# EVA6_Assignments_Session6
EVA6 Assignment for Session 6

The purpose of the assignment is to show the effects of different normalization method in CNN. MNIST data set is used for this purpose. The main objective is to implement below mentioned normalization

   <p align="center" style="padding: 10px">
    <img alt="Forwarding" src="https://github.com/gokul-pv/EVA6_Assignmets_Session6/blob/main/Images/objective.png?raw=true" width =700>
    <br>
    <em style="color: grey">Figure 1.a : Problem Statement</em>
  </p> 
  
The summary of the three different models are shown below

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
