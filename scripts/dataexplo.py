import sys
import io
from BPTorch.datasets import BigPictureRepository, WsiDicomDataset
from contextlib import contextmanager

@contextmanager
def log_to_file(filepath, also_print=True):
    """Captures all stdout/stderr within the block and saves to a file."""
    buffer = io.StringIO()
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    class Tee:
        def write(self, msg):
            buffer.write(msg)
            if also_print:
                original_stdout.write(msg)
        def flush(self):
            buffer.flush()
            if also_print:
                original_stdout.flush()

    sys.stdout = Tee()
    sys.stderr = Tee()
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        with open(filepath, "w") as f:
            f.write(buffer.getvalue())

for p1, p2 in [('fold_0','data/rnd-subset-test'),('fold_1','data/rnd-subset-val'),('fold_2','data/rnd-subsubset-50k')]:
    with log_to_file(f'data/{p1}.txt'):
        print(f'+++++++++++ explo {p1} - {p2} +++++++++++')
        kwargs = WsiDicomDataset.get_default_kwargs()
        ## load trainset and point to patch source
        print(f'+++++++++++ RAW +++++++++++')
        set = BigPictureRepository(f'/mnt/nas6/data/BigPicture_CBIR/datasets/BPTorch/{p1}/BPR.json', load=True, wsidicomdataset_kwargs=kwargs, verbose=False) ## loading valset becuase the content gets overwritten by pointing to preextracted patches. this is just faster than loading the full training fold every time
        print('len:', len(set))
        
        print(f'+++++++++++ SAMPLED +++++++++++')
        set.source_precomputed_patches_from(p2)
        print('len:', len(set))
        print('stats:', set.get_stats())
        set.get_stats_plot(f'data/{p1}')