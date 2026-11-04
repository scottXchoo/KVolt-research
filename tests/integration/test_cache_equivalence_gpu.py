"""Strict native-helper, independent group-axis/bit, GQA kernel diagnostics."""
import pytest

from kvolt_runtime.backend.cache import pack_cache
from kvolt_runtime.qualification.compare import compare_cache

pytestmark=pytest.mark.gpu


@pytest.mark.parametrize('length',[127,128,129,255,256,257,1024,1025,8192])
@pytest.mark.parametrize('group',[8,16,32,64,128])
def test_all_layers_exact_native_helper_and_tail_ownership(gpu_context,length,group):
    torch,helper,quant,_=gpu_context
    generator=torch.Generator(device='cuda').manual_seed(20261104+length)
    # Shared raw fixture storage keeps the 8K diagnostic within AGX memory;
    # both independent implementations must still produce all 32 layer rows.
    key=torch.randn((1,8,length,128),generator=generator,device='cuda',dtype=torch.float16)
    value=torch.randn((1,8,length,128),generator=generator,device='cuda',dtype=torch.float16)
    raw=tuple((key,value) for _ in range(32))
    expected=helper.pack_fp16_cache_bits(raw,group)
    actual=pack_cache(raw,length,group,quantizer=quant)
    torch.cuda.synchronize()
    assert compare_cache(expected,actual,torch) is None
    for row in actual:
        if row[1] is not None:
            assert row[1].is_contiguous() and row[1].untyped_storage().data_ptr()!=key.untyped_storage().data_ptr()
            assert row[1].untyped_storage().nbytes()==row[1].numel()*row[1].element_size()
        assert row[5].is_contiguous() and row[5].untyped_storage().data_ptr()!=value.untyped_storage().data_ptr()
        assert row[5].untyped_storage().nbytes()==row[5].numel()*row[5].element_size()


def independent_word(codes):
    """Python unsigned bit order, independent of Triton/native pack helpers."""
    word=sum(int(code) << (2*index) for index,code in enumerate(codes))
    return word if word < 2**31 else word-2**32


@pytest.mark.parametrize('group',[8,16,32,64,128])
def test_known_quantizer_codes_metadata_group_axes_and_constant(gpu_context,group):
    torch,_,quant,_=gpu_context
    # Min and scale differ across KV head, row and group. Values are exactly
    # representable, so the expected quantized codes require no tolerance.
    values=[]; expected_words=[]; expected_scale=[]; expected_min=[]
    for head in range(8):
        rows=[]; words=[]; scales=[]; minima=[]
        for row in range(128):
            data=[]; scale_row=[]; min_row=[]
            for start in range(0,128,group):
                scale=1 if (start//group)%2==0 else 2
                minimum=(head*4+row%4+start//group)*4
                data.extend(minimum+scale*(position%4) for position in range(group))
                scale_row.append(scale); min_row.append(minimum)
            rows.append(data); words.append([independent_word([(index%4) for index in range(start,start+16)]) for start in range(0,128,16)])
            scales.append(scale_row); minima.append(min_row)
        values.append(rows); expected_words.append(words); expected_scale.append(scales); expected_min.append(minima)
    data=torch.tensor([values],device='cuda',dtype=torch.float16)
    code,scale,minimum=quant(data,group,2)
    assert torch.equal(code,torch.tensor([expected_words],device='cuda',dtype=torch.int32))
    assert torch.equal(scale,torch.tensor([expected_scale],device='cuda',dtype=torch.float16))
    assert torch.equal(minimum,torch.tensor([expected_min],device='cuda',dtype=torch.float16))
    constant=torch.full_like(data,5)
    _,constant_scale,constant_min=quant(constant,group,2)
    assert torch.count_nonzero(constant_scale).item()==0
    assert torch.equal(constant_min,torch.full_like(constant_min,5))
    # Explicit K transpose vs V head-dimension grouping: same fixture viewed
    # along the other axis produces different group minima, checked directly.
    _,kscale,kmin=quant(data.transpose(2,3).contiguous(),group,2)
    grouped=data.transpose(2,3).reshape(1,8,128,128//group,group)
    assert torch.equal(kmin,grouped.amin(-1))
    assert torch.equal(kscale,(grouped.amax(-1)-grouped.amin(-1))/3)


@pytest.mark.parametrize('group',[8,16,32,64,128])
def test_actual_cuda_kernel_uses_each_group_and_gqa_head_exactly(gpu_context,group):
    torch,_,_,matmul=gpu_context
    # One-hot products have one nonzero term. Integer/power-of-two metadata
    # makes FP32 dequant+matmul and the native CUDA output exactly equal.
    words=[]; scales=[]; minima=[]; dequant=[]
    for head in range(8):
        hwords=[]; hscales=[]; hmins=[]; hdequant=[]
        for row in range(128):
            codes=[(index+row+head)%4 for index in range(128)]
            scale=[1 if g%2==0 else 2 for g in range(128//group)]
            minimum=[(head*16+row%8+g)*2 for g in range(128//group)]
            hwords.append([independent_word(codes[start:start+16]) for start in range(0,128,16)])
            hscales.append(scale); hmins.append(minimum)
            hdequant.append([codes[index]*scale[index//group]+minimum[index//group] for index in range(128)])
        words.append(hwords); scales.append(hscales); minima.append(hmins); dequant.append(hdequant)
    code=torch.tensor([words],device='cuda',dtype=torch.int32)
    scale=torch.tensor([scales],device='cuda',dtype=torch.float16)
    minimum=torch.tensor([minima],device='cuda',dtype=torch.float16)
    unpacked=torch.tensor([dequant],device='cuda',dtype=torch.float32).repeat_interleave(4,dim=1)
    # Every output column is examined, including both halves of each G8 word.
    for selected in [0,7,8,15,16,63,127]:
        query=torch.zeros((1,32,1,128),device='cuda',dtype=torch.float16)
        query[:,:,:,selected]=1
        expected=torch.matmul(query.float(),unpacked).half()
        actual=matmul(group,query,code,scale,minimum,2)
        assert torch.equal(expected,actual), {'group':group,'selected':selected}
    # Wrong GQA head and G8 metadata aliases demonstrably change expectations.
    assert not torch.equal(unpacked[:,0],unpacked[:,4])
    if group==8:
        assert not torch.equal(unpacked[:,:,:,0:8],unpacked[:,:,:,8:16])


@pytest.mark.parametrize('length',[127,128,129,255,256,257,1024,1025,8192])
@pytest.mark.parametrize('group',[8,16,32,64,128])
def test_actual_native_attention_residual_update(gpu_context,gpu_attention,length,group):
    """Artificial KV exercises every residual edge without overriding EOS.

    This is a one-attention kernel diagnostic, not a model generation result.
    Full model natural-EOS equivalence remains the separate frozen cohort run.
    """
    torch,helper,quant,_=gpu_context
    attention=gpu_attention
    attention.group_size=attention.config.group_size=group
    generator=torch.Generator(device='cuda').manual_seed(731+length)
    key=torch.randn((1,8,length,128),generator=generator,device='cuda',dtype=torch.float16)
    value=torch.randn((1,8,length,128),generator=generator,device='cuda',dtype=torch.float16)
    raw=tuple((key,value) for _ in range(32))
    expected=helper.pack_fp16_cache_bits(raw,group)
    actual=pack_cache(raw,length,group,quantizer=quant)
    hidden=torch.randn((1,1,4096),generator=generator,device='cuda',dtype=torch.float16)
    position=torch.tensor([[length]],device='cuda',dtype=torch.long)
    with torch.inference_mode():
        native_output,_,native_cache=attention(hidden,position_ids=position,past_key_value=expected[0],use_cache=True)
        candidate_output,_,candidate_cache=attention(hidden,position_ids=position,past_key_value=actual[0],use_cache=True)
    torch.cuda.synchronize()
    assert torch.equal(native_output,candidate_output)
    assert compare_cache((native_cache,),(candidate_cache,),torch) is None
    from kvolt_runtime.backend.cache import validate_packed_cache
    assert validate_packed_cache(tuple(candidate_cache for _ in range(32)),length+1,group)=='cuda:0'
    assert candidate_cache[8]==length+1
    if length%128==127:
        assert candidate_cache[1] is None  # native K residual flushed exactly128
    else:
        assert candidate_cache[1].shape[-2]==(length+1)%128
    assert candidate_cache[5].shape[-2]==128  # V evicts one after residual128
