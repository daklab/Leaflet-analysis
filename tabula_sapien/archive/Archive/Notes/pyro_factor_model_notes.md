## pyro factor model notes 

#### 1. Original code (unoptimized) results for mouse brain simulations ; tabula_muris_brain/Leaflet/run_factor_pyro_model_TM_SS2_simulated.ipynb

Cell types used: Brain_Myeloid_microglial_cell and Brain_Non-Myeloid_oligodendrocyte (actually too slow on these cell types so try two other ones)   

Number of cells: 5968

Number of junctions: 4824

Number of clusters: 1608

# 

Cell types used: 'Brain_Non-Myeloid_neuron', 'Brain_Non-Myeloid_brain_pericyte'

Number of cells: 437

Number of junctions: 3762

Number of clusters: 1254

![Alt text](image-6.png)

![Alt text](image-8.png)

# 

#### 2. Modified code results for mouse brain simulations 

Cell types used: 'Brain_Non-Myeloid_neuron', 'Brain_Non-Myeloid_brain_pericyte'

Number of cells: 437

Number of junctions: 3762

Number of clusters: 1254


![Alt text](image-9.png)

![Alt text](image-10.png)


# 

#### 3. Modified code results for mouse muscle tissue simulations 



#


#### 4. Modified code results for real mouse brain data

Cell types used: All eight 

Number of cells: 5968

Number of junctions: 4824

Number of clusters: 1608

![Alt text](image-2.png)
![Alt text](image-5.png)
![Alt text](image-4.png)
![Alt text](image-12.png)
![Alt text](image-13.png)
![Alt text](image-14.png)
![Alt text](image-17.png)
![Alt text](image-18.png)
![Alt text](image-19.png)
![Alt text](image-21.png)


# 

#### 5. Modified code results for real mouse muscle data

![Alt text](image-11.png)
![Alt text](image-22.png)
![Alt text](image-23.png)
![Alt text](image-24.png)
![Alt text](image-25.png)
![Alt text](image-26.png)

If we run PCA on standradized factor assignment matrix 
![Alt text](image-27.png)
- we see a few outliers 
- not sure where they are coming from
- try a different random seed, how similar are the results and is it the same points that come out as outliers, maybe gets stuck in some local optima where cells are represtned as outliers 

We can also plot the max factor value for each cell where we see in this case compared to mouse brain, many more values are > 0.9 suggesting that the learned factors are more discrete in this case 
![Alt text](image-28.png)

What do the results look like for basic mixture model in this case? do the assignments overlap between the two methods? mixture model doesn't seem to work on this particular dataset in terms of even assigning cells to the same cell states across runs that often because they are probably not specific enough in terms of splicing pattern alone 

#

## Multi-modal autoencoder via splicing and total gene expression 

- In a multi-modal autoencoder, both gene expression and splicing data are treated as input modalities, and the model learns to encode and decode each modality separately. 
- The encoder network maps each modality to a shared latent space, and the decoder network reconstructs each modality from this shared representation.
- In a traditional factor model, you may have separate factors or latent variables for each modality, and the relationship between the observed data and the latent factors can be linear or follow specific probabilistic models.



#

## scVI model 
<img src="image-1.png" alt="Alt text" style="width: 50%;">

- a, The neural networks used to compute the embedding and the distribution of gene expression. NN, neural network. fw and fh are functional representations of NN5 and NN6, respectively. 
- scVI provides a parametric distribution designed to decouple biological signal from the effects of sample-level categorical nuisance factors such as batch annotations and variation in sequencing depth.