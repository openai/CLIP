# The Country211 Dataset

In the paper, we used an image classification dataset called Country211, to evaluate the model's capability on geolocation. To do so, we filtered the YFCC100m dataset that have GPS coordinate corresponding to a [ISO-3166 country code](https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes) and created a balanced dataset by sampling 150 train images, 50 validation images, and 100 test images images for each country.

The following command will download an 11GB archive countaining the images and extract into a subdirectory `country211`:

```bash
wget https://openaipublic.azureedge.net/clip/data/country211.tgz
tar zxvf country211.tgz
```

These images are a subset of the YFCC100m dataset. Use of the underlying media files is subject to the Creative Commons licenses chosen by their creators/uploaders. For more information about the YFCC100M dataset, visit [the official website](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/).