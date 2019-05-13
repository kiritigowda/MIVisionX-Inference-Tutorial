## MIVisionX Inference Tutorial 

<p align="center"><img width="80%" src="images/modelCompilerWorkflow.png" /></p>

In this tutorial, we will learn how to run inference efficiently using [OpenVX](https://www.khronos.org/openvx/) and [OpenVX Extensions](https://www.khronos.org/registry/OpenVX/extensions/vx_khr_nn/1.2/html/index.html). The tutorial will go over each step required to convert a pre-trained neural net model into an OpenVX Graph and run this graph efficiently on any target hardware. In this tutorial, we will learn about AMD MIVisionX which delivers open source implementation of OpenVX and OpenVX Extensions along with MIVisionX Neural Net Model Compiler & Optimizer.

[Neural Net Model Compiler & Optimizer](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/master/model_compiler#neural-net-model-compiler--optimizer) converts pre-trained neural network models to MIVisionX runtime code for optimized inference.

<p align="center"><img width="100%" src="images/frameworks.png" /></p>

Pre-trained models in [ONNX](https://onnx.ai/), [NNEF](https://www.khronos.org/nnef), & [Caffe](http://caffe.berkeleyvision.org/) formats are supported by the model compiler & optimizer. The model compiler first converts the pre-trained models to AMD Neural Net Intermediate Representation (NNIR), once the model has been translated into AMD NNIR (AMD's internal open format), the Optimizer goes through the NNIR and applies various optimizations which would allow the model to be deployed on to target hardware most efficiently. Finally, AMD NNIR is converted into OpenVX C code, which could be compiled and deployed on any targeted hardware.

### Prerequisites

* Ubuntu `16.04`/`18.04` or CentOS `7.5`/`7.6`
* [ROCm supported hardware](https://rocm.github.io/ROCmInstall.html#hardware-support) 
	* GPU or APU required
* [ROCm](https://github.com/RadeonOpenCompute/ROCm#installing-from-amd-rocm-repositories)
* Build & Install [MIVisionX](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX#linux-1)
	* MIVisionX installs model compiler at `/opt/rocm/mivisionx`

## Usage

### Neural Net Model Compiler & Optimizer - OpenVX Code Generation

Use MIVisionX [Neural Net Model Compiler & Optimizer](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/master/model_compiler#neural-net-model-compiler--optimizer) to generate OpenVX code from your pre-trained neural net model. The model compiler generates annmodule.cpp & annmodule.h during the OpenVX code generation. Copy annmodule.cpp & annmodule.h into the module_files folder of this project. The whole process of inference from a pre-trained neural net model will be shown in 3 different samples [below](#sample-1---pre-trained-caffe-model).

### Build - Inference Application

<p align="center"><img width="50%" src="images/app-control.png" /></p>

Once the OpenVX code is generated and the files (annmodule.cpp & annmodule.h) copied into this project folder, follow the instructions below to build the project.

* Clone this Project
````
git clone https://github.com/kiritigowda/MIVisionX-Inference-Tutorial.git
````
* Copy the files (annmodule.cpp & annmodule.h) generated by the model compiler into this project's module_files folder.

* Build this project
````
cd MIVisionX-Image-Classifier
mkdir build
cd build
cmake ../
make
````

### Run

<p align="center"><img width="50%" src="images/app_display.png" /></p>

```
./classifier	--mode				<1/2/3 - 1:classification 2:detetction 3:segmentation>	[required]
		--video/--capture/--image	<video file>/<0>/<image file>				[required]
		--model_weights			<model_weights.bin>					[required]
		--label				<label text>						[required]
		--model_inputs			<c,h,w - channel,height,width>				[required]
		--model_outputs			<c,h,w - channel,height,width>				[required]

		--model_name			<model name>					[optional - default:NN_ModelName]
		--add				<Ax,Ay,Az - input preprocessing factor>		[optional - default:0,0,0]
		--multiply			<Mx,My,Mz - input preprocessing factor>		[optional - default:1,1,1]


[usage help]	--help/--h

```
### Tested Models
* [GoogleNet](http://www.cs.bu.edu/groups/ivc/data/SOS/GoogleNet_SOS.caffemodel)
* [InceptionV4](https://github.com/soeaver/caffe-model/tree/master/cls#performance-on-imagenet-validation)
* [ResNet50](https://github.com/KaimingHe/deep-residual-networks#deep-residual-networks)
* [ResNet101](https://github.com/KaimingHe/deep-residual-networks#deep-residual-networks)
* [ResNet152](https://github.com/KaimingHe/deep-residual-networks#deep-residual-networks)
* [VGG16](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel)
* [VGG19](http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel)


#### Generating weights.bin, annmodule.cpp, & annmodule.h for different Models

1. Download or train your own caffemodel for the supported models listed above.

2. Use [MIVisionX Model Compiler](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX/tree/master/model_compiler#neural-net-model-compiler--optimizer) to generate OpenVX C Code from the pre-trained caffe models.

**Note:** MIVisionX installs all the model compiler scripts in `/opt/rocm/mivisionx/model_compiler/python/` folder


* Convert the pre-trained caffemodel into AMD NNIR model:

	````
	% python /opt/rocm/mivisionx/model_compiler/python/caffe_to_nnir.py <net.caffeModel> <nnirOutputFolder> --input-dims <n,c,h,w> [--verbose <0|1>]
	````
		Sample:
		% python /opt/rocm/mivisionx/model_compiler/python/caffe_to_nnir.py VGG_ILSVRC_16_layers.caffemodel VGG16_NNIR --input-dims 1,3,224,224

* Convert an AMD NNIR model into OpenVX C code:

	````
	% python /opt/rocm/mivisionx/model_compiler/python/nnir_to_openvx.py <nnirModelFolder> <nnirModelOutputFolder>
	````
		Sample:
		% python /opt/rocm/mivisionx/model_compiler/python/nnir_to_openvx.py VGG16_NNIR VGG16_OpenVX

**Note:**

	* The weights.bin, annmodule.cpp, & annmodule.h files will be generated inside the OpenVX folder and you can use these as inputs for this project.
	* Copy annmodule.cpp & annmodule.h into the module_files folder.

#### label < path to labels file >

Use [labels.txt](data/labels.txt) or [simple_labels.txt](data/simple_labels.txt) file in the data folder

#### video < path to video file >

Run classification on pre-recorded video with this option.

#### capture <0>

Run classification on the live camera feed with this option.

**Note:** --video and --capture options are not supported concurrently

# Supported Pre-Trained Model Formats
* Caffe
* NNEF
* ONNX

<p align="center"><img width="70%" src="images/modelTrainedFrameWorks.png" /></p>

## Sample 1 - Pre-Trained Caffe Model

### Run VGG 16 Classification on Live Video

<p align="center"><img width="50%" src="images/app-control.png" /></p>

* **Step 1:** Install all the Prerequisites

	**Note:** MIVisionX installs all the model compiler scripts in `/opt/rocm/mivisionx/model_compiler/python/` folder

* **Step 2:** Download pre-trained VGG 16 caffe model - [VGG_ILSVRC_16_layers.caffemodel](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel)


* **Step 3:** Use MIVisionX Model Compiler to generate OpenVX files from the pre-trained caffe model


	* Convert .caffemodel to NNIR

	````
	% python /opt/rocm/mivisionx/model_compiler/python/caffe_to_nnir.py upgraded_VGG_ILSVRC_16_layers.caffemodel VGG16_NNIR --input-dims 1,3,224,224
	````

	* Convert NNIR to OpenVX

	````
	% python /opt/rocm/mivisionx/model_compiler/python/nnir_to_openvx.py VGG16_NNIR VGG16_OpenVX
	````

	**Note:** 

	* Copy annmodule.cpp & annmodule.h into the module_files folder 
	* After copying the files, build this project
	* Use weights.bin generated in VGG16_OpenVX folder for the classifier --model_weights option
	
	<p align="center"><img width="50%" src="images/app_display.png" /></p>

```
./classifier    --label PATH_TO/data/simple_labels.txt 
                --capture 0 
                --model_weights PATH_TO/VGG16_OpenVX/weights.bin
		--model_name VGGNet-16
		--model_inputs 3,224,224
		--model_outputs 1000
```
