import tensorflow as tf


def get_layouts(func, layouts=[]):
    for op in func.graph.get_operations():
        node = op.node_def
        if 'data_format' in node.attr:
            layout = node.attr['data_format'].s.decode('utf-8') # NHWC or NCHW
            layouts.append((op, layout))

    # Go through sub-functions
    subfuncs = func.graph._functions.values() # please comment if there's a better way to obtain this list than by accessing the internal-dict
    for func in subfuncs:
        get_layouts(func, layouts)
    
    return layouts


def main():
    model_dir = "/work/rwth0792/fortran-ml-interface/model/unet/Model_1251_loss_0.00012877.tf"
    model = tf.saved_model.load(model_dir)
    sig = model.signatures['serving_default']
    graph_def = sig.graph.as_graph_def()

    layouts = get_layouts(sig)
    print(layouts)


if __name__ == "__main__":
    main()