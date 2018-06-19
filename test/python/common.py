import platform

import tensorflow as tf

class NgraphTest(object):
    test_device = "/device:NGRAPH:0"
    soft_placement = False
    log_placement = True

    @property
    def config(self):
        return tf.ConfigProto(
            allow_soft_placement=self.soft_placement,
            log_device_placement=self.log_placement,
            inter_op_parallelism_threads=1)
