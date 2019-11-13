## **Instruction**
Check  
* 'Semantic Instance Segmentation with a Discriminative Loss Function'
* 'ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation'
* 'Towards End-to-End Lane Detection: an Instance Segmentation'


## **File information**

* `instance_seg_models_enet_train.py`  
Main train model script.

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
the output(predict_train) images will be saved in **_A1_predict_XXX** folder.
~~~
'xxx' means architecture of models.
 -deeplabv1, enet, etc..
~~~

* `semantic_seg_apply_crf.py`  
crf applied images in _**A1_predict**_ and save image into **_A1_predict_crf_**


<h2>Data folder information</h2>

* `A1`  
This folder is original Data set for Instance segmentation.

~~~
_centers.png : Center of each leaf.
_fg.png : Sementic segmentation label.
_rgb.png : instance segmentation label.
~~~


 

