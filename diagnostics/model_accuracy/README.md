# Verify model accuracy by running the models using ngraph

This model_accuracy tool will run inference for Image Recognition models - Inception_v4, ResNet50_v1, Mobilenet_v1.(The toold will be extended to run more models).
After running the models using ngraph, the tool validates the accuracy by comparing with the known accuracy from published papers.

# Required setup to use the tool:


# To run the model test tool(example):
	python verify_inference_model.py --model_name Inception_v4

# Limitations:
The tool can run only one network at a time now, will be extended to run multiple models at once and validate accuracy of the same 
