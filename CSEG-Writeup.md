# Geochemical Facies Analysis using Machine Learning

Brendon Hall, [Enthought](https://www.enthought.com/)

The practice of extracting insights from large datasets has certainly caught mainstream attention in recent years.  Terms like 'Big Data' and 'multivariate statistics' pervade industry literature and conference show floors.  Geoscientists, however, have been wrangling massive multivariate datasets since the first seismic survey. While there have been recent advances in machine learning (eg. deep neural networks) many of the multivariate approaches to learning from data have been around for a long time. There are also a plethora of libraries available that implement these methods.  One of the most popular libraries is `scikit-learn` (http://scikit-learn.org/stable/), a collection of machine learning libraries in Python.  It contains many tools for data mining and analysis including classification, clustering and regression algorithms.  

In this tutorial we demonstrate how dimensionality reduction and unsupervised machine learning can be used to analyze X-ray flourescence measurements of cutting samples.  Dimensionality reduction is the process of reducing the number of random variables under consideration by obtaining a smaller set of variables that are better at describing the variation within the dataset. Clustering is an unsupervised machine learning technique that learns an optimal grouping from the data itself and doesn't require training data.  The groups consist of samples with similar characteristics, which can be considered as distinct geochemical facies.  This will be implemented using Python, the `scikit-learn` library and other popular Python data science tools. 

```python
import pandas as pd
import numpy as np

# Machine learning libraries
from sklearn.preprocessing import scale
from sklearn.decomposition import FactorAnalysis
from sklearn.cluster import KMeans

# Visualization libraries
import matplotlib.pylab as plt
import seaborn as sns
```

## XRF Cuttings Analysis

X-ray flourescence (XRF) is becoming a common wellsite cuttings analysis technique (Carr et al, 2014).  Portable XRF provide rapid elemental composition measurements of cuttings as they are drilled.  These devices measure the flourescent x-rays emitted by a sample when illuminated by an energetic X-ray source.  Elements in the sample will emit X-rays certain specific wavelengths, and the spectrum of the emitted x-rays can be measured and used to quantify the amount of the corresponding elements in present in the sample.  Trends in element concentration can be use to infer sediment depositional environment, sources, and indicate conditions conducive to the preservation of organic material. XRF data can be used to for geologic characterization, optimize well placement and provide additional guidance for geosteering.

The data for this tutorial consists of XRF measurements of cuttings from the lateral section of an unconventional well.  The cuttings measurements are made at approximately 10m intervals.  In this case the raw data consists of 22 measurements for each sample.  Each measurement gives the weight percentage of a component.  The data is read from a `csv` file into a dataframe using the `pandas` library, which provides many convenient data structures and tools for data science. 

```python
geochem_df = pd.read_csv('XRF_dataset.csv')
```

In machine learning, the term *feature* is used to refer to the attributes of the objects being considered that can be used to describe, cluster and classify them.  In this case, the objects being studied are cuttings samples, and the features are the 22 XRF measurements.  We can use *feature engineering* to augment this dataset.  This refers to the process of using domain knowledge to create new features that help machine learning algorithms discriminate the data.  

In geochemistry, elements are used as proxies that give hints to the physical, chemical or biological events that were occuring during its formation.  Ratios of certain elements can indicate the relative strength of various effects.  For example, The Si/Zr ratio can used to record high biogenic silica relative to terrestrial detrital input (associated with sandstone and siltstone).  The Si/Al ratio is used as a proxy for biogenic silica to aluminous clay (Croudace and Rothwell, 2015).  The Zr/Al ratio proxy for terrigenous input, geochemical behavior of Zr suggests that this ratio can be used as a proxy for grain size (Calvert and Pederson, 2007).

```python
geochem_df['Si/Zr'] = geochem_df['SiO2'] / geochem_df['Zr']
geochem_df['Si/Al'] = geochem_df['SiO2'] / geochem_df['Al2O3']
geochem_df['Zr/Al'] = geochem_df['Zr'] / geochem_df['Al2O3']
```

## Dimensionality Reduction

Not surprisingly, multivariate datasets are characterized by the fact that they contain plenty of variables.  This richness can be used to explain complex behaviour that can't be captured with a single observation.  Multivariate methods allow us to consider changes in several observations simultaneously.  With many observations it is quite likely that the changes we observe are related to a smaller number of underlying causes.  Dimensionality Reduction is the process of using the correlations in the observed data to reveal a more parsimonious underlying model that explains the variation in the observed data.

Exploratory factor analysis (EFA) reduces the number of variables by identifying the underlying *latent factors* present in the dataset.  These factors cannot be measured directly, can be determined only by measuring manifest properties.  For example, in the case of our geochemical dataset, a *shaliness* factor could be associated with high XX and XX readings. EFA assumes that the observations are a linear combination of the underlying factors, plus some Gaussian noise.    A related dimensionality reduction technique is principle component analysis (PCA).  In EFA, the factors are responsible for causing the responses in the measured variables.  PCA determines components that are weighted linear combinations of the observations. 

Before applying EFA, the dataset should be standardized.  If the measurements were made using different scales, this can affect the weights assigned to each factor.  This preprocessing operation rescales each variable to have zero mean and unit variance. 

```python
from sklearn.preprocessing import scale
data = geochem_df.ix[:, 2:]
data = scale(data)
```

EFA requires that the number of factors to be extracted is specified *a-priori*.  It is often not immediately obvious how many factors should be specified.  Many authors have proposed rules over the years (eg. Preacher et al, 2013).  One simple approach (known as the Kaiser criterion) involves looking at the eigenvalues of the covariance matrix of the data, and counting the number above a threshold (typically 1.0).  In Figure 1, there are 6 eigenvalues greater than 1.0 (dashed red line), suggesting there are 6 relevant factors to be extracted.

![Crossplot Data](images/1_eigs.png)
**Figure 1** Eigenvalues of covariance matrix. 

The scikit-learn library contains a FactorAnalysis module that can be used to extract the 6 factors.  This is done by creating a factor analysis object and fitting the model to the data.

```python
fa_model = FactorAnalysis(n_components = 6)
fa_model.fit(data)
factor_data = fa_model.transform(data)
```

### Interpreting the factors

The factors can now be examined to interpret the underlying properties they represent.  Figure X shows the *factor loadings* associated with the first factor.  The loading score indicates the correlation between the factor and the observed variable.  In this case, the first factor is associated with high values of Calcite, Dolomite and CaO.  We could interpret this factor as representing the carbonate character of the rock.  Similar interpretations can be given to the other factors by observing their loading scores.

![Factor Loadings](images/2_Factor_interp.png)
**Figure 2** Factor loadings associated with the 4th extracted factor.

## Clustering
   
The factor analysis has reduced the initial collection of 25 XRF features in a reduced set of 7 factors that account for most of the variation in the data.  A logical next step would be using these factors to group the cutting samples by their common geochemical traits, or *geochemical facies*.  Cluster analysis is a suitable approach for assigning a common facies label to similar samples. Clustering attempts to group samples so that those in the same group (or cluster) are more similar than those in other clusters.  Cluster analysis is one class of techniques that fall under the category of *unsupervised* machine learning.  These approaches are used to infer structure from the data itself, without the use of labeled training data to guide the model. 

The [K-Means](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans) algorithm clusters data by trying to separate samples in $n$ groups of equal variance.  The algorithm locates the optimal cluster centroids by minimizing the distance between each point in a cluster and the closest centroid. The algorithm has three steps.  It initializes by picking locations for the initial $n$ centroids (often random samples from the dataset).  Next, each sample is assigned to one of the $n$ groups according to the nearest centroid.  New centroids are then calculated by finding the mean values of each sample in each group.  This is repeated until the difference between subsequent centroid positions falls below a given threshold.

Similar to EFA, K-Means requires that the number of clusters be specified before running the algorithm.  There are a number of approaches to finding the optimal number of clusters.  The goal is to choose the minimum number of clusters that accurately partition the dataset.  These range from the relatively simple 'elbow method' to more rigorous techniques involving the Bayesian information criterion and optimizing the Gaussian nature of each cluster (Hamerly and Elkan, 2003).  Figure 3 demonstrates the 'elbow method' applied to this dataset.  The sum of the squared distance of each point to the nearest cluster centroid is plotted for an increasing number of clusters.  As the number of clusters is increased, the error decreases as the clusters better fit the data. The elbow of the curve represents the point of diminishing returns where increasing the number of clusters doesn't reduce the error appreciably.  Figure 3 suggests that about 7 clusters would be adequate for this dataset. 

![Model Selection](images/3_k_means.png)
**Figure 3** Mean squared error vs. number of clusters for the XRF dataset.

The K-means algorithm in `scikit-learn` is used to cluster the reduced dataset.  Similar to the factor analysis, this is done by creating a K-means model and fitting the factor dataset.

```python
kmeans = KMeans(n_clusters=7, random_state=0)
kmeans.fit(factor_data)
```

### Interpreting the clusters

Each sample in the dataset has now been assigned to one of seven clusters.  If we are going  to interpret these clusters as geochemical facies, it is useful to inspect the to inspect the geochemical signature of each cluster. Figure 4 contains a series of box plots that show the distribution of a small subset of measurements across each of the 7 clusters.  Box plots depict 5 descriptive statistics; the horizontal lines of the colored rectangle show the first quartile, median and third quartile.  The arms show the minimum and maximum.  Outliers are shown as black diamonds.  This plot is generated using the statistical visualization library `seaborn` [https://seaborn.pydata.org/](https://seaborn.pydata.org/).

Figure 4A indicates that Cluster 2 is characterized by a relatively high (and variable) Si/Zr ratio.  Cluster 4 has a high Zr/Al ratio (4B, 4C) and cluster 3 has a high MgO signature.  This can be done for each measurement to build up a geologic interpretation of each cluster.

![Cluster fingerprint](images/4_Cluster_fingerprint.png)
**Figure 4** Distribution of calcite across the clusters.

## Visualizing results

Now we have organized every cutting measurement into 7 geochemical facies (clusters), we can visualize the classification in a log plot to better understand how the facies transition to one another in the context of a well.  The right column of Figure 5 shows the clusters assigned to each sample using a unique color, indexed by measured depth (MD).  The other columns show 4 of the corresponding geochemical measurements.  Similar plots could be made for the other wells in the dataset and used to identify common intervals.

![Log plot](images/5_logs.png)
**Figure 5** Log style plot showing 4 geochemical measurements and cluster assigments.

This analysis provides data that can be used for geosteering horizontal wells.  This is useful in areas that lack a distinctive gamma ray signature. Classifing the geochemical facies of a cuttings sample can be used to help pinpoint the location of the well given an exisiting chemo-stratigraphic framework.  To build up this framework, it is helpful to plot the geochemical facies along the well path.  Figure 6 shows the trajectory (TVD vs. MD) for Well 1, with the different facies colored using the same scheme as Figure 5.  This can be used to build a psuedo-vertical profile and help identify specific zones as the well porpoises up and down along its length.

![Well trajectory](images/6_Well_trajectory.png)
**Figure 6** Geochemical facies assignments plotted along well trajectory.

This tutorial has demonstrated how dimensionality reduction and unsupervised machine learning can be used to understand and analyze XRF measurements of cuttings to determine geochemical facies.  Exploratory factor analysis yields insight into the underlying rock properties that are changing across the reservoir.  K-means clustering is used to organize similar samples into a smaller number of groups that can be interpreted as geochemical facies.  This can be used to correlate formation tops between wells and provide data necessary for geosteering.

The code shown in this article demonstrates how to perform most of this analysis using Python and some common toolboxes.  For the full code used to generate these results, please see the github repo at [www.github.com/brendonhall/clustering_tutorial](www.github.com/brendonhall/clustering_tutorial).

## References

Carr, R., Yarbrough, L., Lentz, N., Neset, K., Lucero, B. and Kirst, T. (2014) *On-Site XRF Analysis of Drill Cuttings in the Williston Basin*. URTeC Denver, Colorado, USA. doi:[10.15530/urtec-2014-1934308](https://www.crossref.org/iPage?doi=10.15530%2Furtec-2014-1934308)

Calvert, S. and Pedersen, T. (2007). *Elemental proxies for paleoclimatic and palaeoceanographic variability in marine sediments: interpretation and application*. Developments in Marine Geology. 1. 568-644.

Croudace, I. W., and R. G. Rothwell. *Micro-XRF Studies of Sediment Cores*. Springer, 2015.

Davis, J. C. *Statistical and Data Analysis in Geology*. J. Wiley, 1986.

Hamerly, G. and Elkan, C. (2003) *Learning the k in k-means*. Advances in Neural Information Processing Systems 16, edited by Thrun, S., Saul, L. K. and Sch&ouml;lkopf, B. MIT Press, 281-288.

Iwamori, H., Yoshida, K., Nakamura, H. Kuwatani, T., Hamada, M., Haraguchi, S. and Ueki, K. (2017) *Classification of geochemical data based on multivariate statistical analyses: Complementary roles of cluster, principal component, and independent component analyses*. Geochemisty, Geophysics, Geosystems. 18(3).  994–1012. doi:[10.1002/2016GC006663](http://onlinelibrary.wiley.com/doi/10.1002/2016GC006663/abstract)

Preacher K.J., Zhang G., Kim C. and Mels G. (2013) *Choosing the Optimal Number of Factors in Exploratory Factor Analysis: A Model Selection Perspective*. Multivariate Behav Res. 48(1).

Waskom, M., et al. (2017) Seaborn, statistical data visualization library. v0.8.1 doi: [10.5281/zenodo.883859](https://doi.org/10.5281/zenodo.883859)








