import numpy as np
import pytest
from benchmarks.tensor_frontier.benchmark_mode_gram_batch import benchmark,check_batch,numpy_checked,checked_build


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),2.])
def test_bad_member_fails(bad):
    with pytest.raises(ValueError,match='tolerance'):
        check_batch([((1,),),((1,),)],np.array([[[1]],[[bad]]]))


def test_wrong_batch_shape_fails():
    with pytest.raises(ValueError,match='shape'):
        check_batch([((1,),),((1,),)],np.ones((1,1,1)))


@pytest.mark.parametrize('counts',[(-1,1,1),(0,0,1),(0,1,0)])
def test_counts_before_loading(counts):
    with pytest.raises(ValueError,match='counts'):
        benchmark('absent','absent','absent',*counts)


def test_numpy_guard_before_output_write():
    x=np.ones((2,2,3,4));x[-1,-1,-1,-1]=np.nan
    output=np.full((2,3,3),42.)
    with pytest.raises(ValueError,match='input'):numpy_checked(x,1,output)
    assert np.all(output==42.)


def test_build_dependency_hash_guard(tmp_path):
    import json
    from benchmarks.tensor_frontier.profile_mode_gram import sha
    lib=tmp_path/'fake.so';lib.write_bytes(b'test')
    lib.with_suffix('.build.json').write_text(json.dumps({'binary_sha256':sha(lib),
        'sources_sha256':{'mojo/tensor_frontier/mode_gram.mojo':'invalid'}}))
    with pytest.raises(ValueError,match='dependency'):checked_build(lib)
