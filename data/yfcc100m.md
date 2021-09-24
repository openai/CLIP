# The YFCC100M Subset

In the paper, we performed a dataset ablation using a subset of the YFCC100M dataset and showed that the performance remained largely similar. 

The subset contains 14,829,396 images, about 15% of the full dataset, which have been filtered to only keep those with natural languag titles and/or descriptions in English.

We provide the list of (line number, photo identifier, photo hash) of each image contained in this subset. These correspond to the first three columns in the dataset's metadata TSV file.

```bash
wget https://openaipublic.azureedge.net/clip/data/yfcc100m_subset_data.tsv.bz2
bunzip2 yfcc100m_subset_data.tsv.bz2
```

Use of the underlying media files is subject to the Creative Commons licenses chosen by their creators/uploaders. For more information about the YFCC100M dataset, visit [the official website](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/).