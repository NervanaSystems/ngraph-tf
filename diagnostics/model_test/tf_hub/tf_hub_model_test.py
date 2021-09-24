import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import ngraph_config
import argparse

url_dict = {}
url_dict[
    "mobilenetv2"] = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/2"
url_dict[
    "inceptionv3"] = "https://tfhub.dev/google/imagenet/inception_v3/classification/1"
url_dict[
    "mobilenetv1"] = "https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/1"
url_dict[
    "inception_resentv2"] = "https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/1"
url_dict[
    "resnetv2"] = "https://tfhub.dev/google/imagenet/resnet_v2_101/classification/1"
url_dict[
    "inceptionv1"] = "https://tfhub.dev/google/imagenet/inception_v1/classification/1"
url_dict[
    "pnasnet"] = "https://tfhub.dev/google/imagenet/pnasnet_large/classification/2"
url_dict[
    "nasnet"] = "https://tfhub.dev/google/imagenet/nasnet_large/classification/1"


def compare(model_name, batch_size, threshold=0.001):
    print("Running {} on Ngraph".format(model_name))
    module_url = url_dict[model_name]
    # Running on NGraph
    with tf.Graph().as_default():
        module = hub.Module(module_url)
        height, width = hub.get_expected_image_size(module)
        np.random.seed(100)
        random_input = np.random.random_sample([batch_size, height, width, 3])
        logits_ngraph = module(random_input)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())

            ngraph_result = sess.run(logits_ngraph)

    ngraph_config.disable()
    if ngraph_config.is_enabled():
        raise Exception("Ngraph should be disabled")
    else:
        print("Running {} on CPU".format(model_name))

    with tf.Graph().as_default():
        module_cpu = hub.Module(module_url)
        logits_cpu = module_cpu(random_input)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            cpu_result = sess.run(logits_cpu)

    l1_norm = np.linalg.norm((ngraph_result - cpu_result), 1)
    l2_norm = np.linalg.norm((ngraph_result - cpu_result), 2)
    inf_norm = np.linalg.norm((ngraph_result - cpu_result), np.inf)
    try: 
        assert l1_norm < threshold
        print ("l1 norm for model {} passed".format(model_name))
    except AssertionError:
        print("l1 norm {} is greater than the threshold {} set".format(l1_norm,threshold))
    try:
        assert l2_norm < threshold
        print ("l2 norm for model {} passed".format(model_name))
    except AssertionError:
        print("l2 norm {0} is greater than the threshold {} set".format(l2_norm,threshold))
    try:
        assert inf_norm < threshold
        print ("inf norm for model {} passed\n".format(model_name))
    except AssertionError:
        print("inf norm {} is greater than the threshold {} set\n".format(inf_norm,threshold))

if __name__ == "__main__":
    classification_model = ["mobilenetv2", "inceptionv3", "mobilenetv1","inception_resentv2","resnetv2","inceptionv1","pnasnet","nasnet"]
    for model in classification_model:
        compare(model, 100)



