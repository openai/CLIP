# The Rendered SST2 Dataset

In the paper, we used an image classification dataset called Rendered SST2, to evaluate the model's capability on optical character recognition. To do so, we rendered the sentences in the [Standford Sentiment Treebank v2](https://nlp.stanford.edu/sentiment/treebank.html) dataset and used those as the input to the CLIP image encoder.

The following command will download a 131MB archive countaining the images and extract into a subdirectory `rendered-sst2`:

```bash
wget https://openaipublic.azureedge.net/clip/data/rendered-sst2.tgz
tar zxvf rendered-sst2.tgz
```

