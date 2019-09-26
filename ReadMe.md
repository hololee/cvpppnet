## **istruction**
check  
* 'Towards End-to-End Lane Detection: an Instance Segmentation'
* 'ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation'
* 'Towards End-to-End Lane Detection: an Instance Segmentation'


## **File information**

* `batch_norm.py`  
class for using batch normalization.

* `config_etc.py`  
for configure some params to train or something.  

* `DataGen.py`  
class for create data batches or load images.

* `method.py`  
methods like convolution function, etc..  

* `placeHolders.py`  
place holder class for some params.
**_using tf.placeholder(...)_**  

* `sementic_seg_models_xxx.py`  
tensorflow train architecture for semantic segmentation.   
this models use _**A1**_(original images) for segmentation.   
the output(predict_train) images will be saved in **_A1_predict_** folder.
~~~
'xxx' means architecture of models.
 -deeplabv1, enet, etc..
~~~

* `semantic_seg_apply_crf.py`  
crf applied images in _**A1_predict**_ and save image into **_A1_predict_crf_**



 

