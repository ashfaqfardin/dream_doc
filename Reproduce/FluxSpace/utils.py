from safetensors import safe_open

def get_attr(obj, attr):
    attrs = attr.split(".")
    for name in attrs:
        obj = getattr(obj, name)
    return obj


def load_from_safetensors(ckpt_path):
    assert ckpt_path.endswith(".safetensors"), "The ckpt should be a safetensors file"
    tensors = {}
    with safe_open(ckpt_path, framework="pt", device=0) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    return tensors

def set_attr_raw(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    setattr(obj, attrs[-1], value)
