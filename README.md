# EVA6_Assignments_Session6
EVA6 Assignment for Session 6

The main objective of this assignment is to implement below mentioned normalization technique on MNIST data and study their characteristics. 

   <p align="center" style="padding: 10px">
    <img alt="Forwarding" src="https://github.com/gokul-pv/EVA6_Assignmets_Session6/blob/main/Images/objective.png?raw=true" width =700>
    <br>
    <em style="color: grey">Figure 1.a : Problem Statement</em>
  </p> 
  
The **summary** of the three different models are shown below

  <p align="center" style="padding: 10px">
    <img alt="Forwarding" src="https://github.com/gokul-pv/EVA6_Assignmets_Session6/blob/main/Images/model_summary_bn.png?raw=true" width =500>
    <br>
    <em style="color: grey">Figure 1.b : Model Summary BN with L1 loss</em>
  </p> 
  
  <p align="center" style="padding: 10px">
    <img alt="Forwarding" src="https://github.com/gokul-pv/EVA6_Assignmets_Session6/blob/main/Images/model_summary_ln.png?raw=true" width =500>
    <br>
    <em style="color: grey">Figure 1.c : Model Summary Layer norm</em>
  </p> 
  
  <p align="center" style="padding: 10px">
    <img alt="Forwarding" src="https://github.com/gokul-pv/EVA6_Assignmets_Session6/blob/main/Images/model_summary_gn.png?raw=true" width =500>
    <br>
    <em style="color: grey">Figure 1.d : Model Summary Group norm</em>
  </p> 


**Code Explanation**

The model.py file contains the class Net which defines the CNN. The class take one argument which defines the type of normalization to be used:

 - 'b' for Batch normalization (default)
 - 'l' for Layer normalization
 - 'g' for Group normalization

To implement L1 loss you have to pass the argument **isL1 = True** while calling the training function **train**. The default value is false.

**Overview of Normalization**

The normalization methods used here are Batch Normalization, Layer Normalization and Group Normalization.

As you can see from the below image write the input image shape as [N, C, H, W]. The main difference between these methods is that

-   Batch Norm is on the batch, normalizing the NHW, is to normalize each single channel input, which is not good for small batch size
-   Layer Norm normalizes the CHW in the channel direction, which is to normalize the input at each depth, mainly for the RNN 
-   Group Norm groups the channels, which is similar to LN, except that GN divides the channels, refines them, and then normalizes them

![enter image description here](https://images3.programmersought.com/664/06/067df05358305059bc6648a612267ec8.png)


**Misclassified Images during validation**

Misclassified images during validation for all the three models are shown below


  <p align="center" style="padding: 10px">
    <img alt="Forwarding" src="https://github.com/gokul-pv/EVA6_Assignmets_Session6/blob/main/Images/Misclassified_bn.png?raw=true" width =1000>
    <br>
    <em style="color: grey">Figure 1.e : Misclassified: BN with L1 loss</em>
  </p> 
  
  
   <p align="center" style="padding: 10px">
    <img alt="Forwarding" src="https://github.com/gokul-pv/EVA6_Assignmets_Session6/blob/main/Images/Misclassified_ln.png?raw=true" width =1000>
    <br>
    <em style="color: grey">Figure 1.f : Misclassified: Layer norm</em>
  </p> 


   <p align="center" style="padding: 10px">
    <img alt="Forwarding" src="https://github.com/gokul-pv/EVA6_Assignmets_Session6/blob/main/Images/Misclassified_gn.png?raw=true" width =1000>
    <br>
    <em style="color: grey">Figure 1.g : Misclassified: Group Norm</em>
  </p> 

**Result**

The test and validation loss and accuracy are shown below

 <p align="center" style="padding: 10px">
    <img alt="Forwarding" src="https://github.com/gokul-pv/EVA6_Assignmets_Session6/blob/main/Images/loss_accurarcy_plot.png?raw=true" width =700>
    <br>
    <em style="color: grey">Figure 1.h : Loss and accuracy plots</em>
  </p>

