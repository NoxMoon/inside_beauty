<center><h1> Inside Beauty</h1></center>

This repository is my first capstone project for Springboard data science career track. I explored ∼8000 cosmetic products, with information of brand, category, ingredients, packaging scraped from cosmetic review website [beautypedia.com](https://www.beautypedia.com). I am particulaly interested in using ingredient information to work on the following two problems that can help people understand cosmetic products from a chemical aspect:
- Classify product categories using ingredient-related features. Identify key ingredients for different product
categories. Find categories that are similar in formula (e.g. eye creams and moisturizers) that customers
may consider using in replacement of each other.
- Predict the price of products. Assess the relative importance of ingredients in determining price versus
other factors such as brand and packaging.

Final report of this project can be found [here](https://github.com/NoxMoon/inside_beauty/blob/master/documents/final%20report.ipynb)

## Table of content
* Data Acquisition
[beautypedia_scraper.ipynb](https://github.com/NoxMoon/inside_beauty/blob/master/web_scraper/beautypedia_scraper.ipynb),
[beautypedia_ingredient_scraper.ipynb](https://github.com/NoxMoon/inside_beauty/blob/master/web_scraper/beautypedia_ingredient_scraper.ipynb)
* Data Cleaning
    * Ingredient Matching
    * Further Cleaning and Feature Engineering
    [image_preprocessing.ipynb](https://github.com/NoxMoon/inside_beauty/blob/master/data_cleaning/image_preprocessing.ipynb), 
[logo_image_filter.ipynb](https://github.com/NoxMoon/inside_beauty/blob/master/data_cleaning/logo_image_filter.ipynb)
    * Image Preprocessing and Logo Image Filtering
    [image_preprocessing.ipynb](https://github.com/NoxMoon/inside_beauty/blob/master/data_cleaning/image_preprocessing.ipynb), 
[logo_image_filter.ipynb](https://github.com/NoxMoon/inside_beauty/blob/master/data_cleaning/logo_image_filter.ipynb)
* Exploratory Data Analysis
    * Visualization
    [exploratory_data_analysis.ipynb](https://github.com/NoxMoon/inside_beauty/blob/master/eda/exploratory_data_analysis.ipynb)
    * Statistical testing 
    [statistical_test_price.ipynb](https://github.com/NoxMoon/inside_beauty/blob/master/eda/statistical_test_price.ipynb)
* Machine Learning
    * tSNE with ingredient features
    [tSNE.ipynb](https://github.com/NoxMoon/inside_beauty/blob/master/machine_learning/tSNE.ipynb)
    * Product category classification with ingredient features
    [product_category_classification.ipynb](https://github.com/NoxMoon/inside_beauty/blob/master/machine_learning/product_category_classification.ipynb)
    * Price Regression 
    [price_regression.ipynb](https://github.com/NoxMoon/inside_beauty/blob/master/machine_learning/price_regression.ipynb)
[packaging_to_price.ipynb](https://github.com/NoxMoon/inside_beauty/blob/master/machine_learning/packaging_to_price.ipynb)
* Conclusion

**Packages used**: numpy, scipy, statsmodel, scikit-learn, pandas, lightgbm, pytorch, skimage, seaborn, matplotlib, BeautifulSoup, selenium.

## Data Acquisition

Product and ingredient information are scraped from [Beautypedia](https://www.beautypedia.com) and [Paula’s choice](https://www.paulaschoice.com/ingredient-dictionary) websites. These are websites run by Paula Begoun and her team, where they constantly post reviews on cosmetic products.
* Product information. 
    * name: product name 
    * category: subcategory of products 
    * brand: product brand 
    * ingredient: list of ingredients in a product 
    * image: product image

We created three main tables for skin care, body care and makeup products. Products are further divided into subcategories such as moisturizer, serum, sunscreen, exfoliator…, and one product may belong to multiple categories. There are 4810, 419, 2513 unique products for skin care, body care and makeups, respectively. 

* Ingredient information of 1750 ingredients.
    * name: ingredient name 
    * rating: rating of each ingredient according to Paula and her team 
    * category: ingredient category -- indicate an ingredient’s function in products. An ingredient may belong to several categories.
    
Webscrapers can be found here: [beautypedia_scraper.ipynb](https://github.com/NoxMoon/inside_beauty/blob/master/web_scraper/beautypedia_scraper.ipynb),
[beautypedia_ingredient_scraper.ipynb](https://github.com/NoxMoon/inside_beauty/blob/master/web_scraper/beautypedia_ingredient_scraper.ipynb)


## Data Cleaning

#### Ingredient Matching
Different companies may list the same ingredient in different ways. The most common ingredient water, can appear as “water”, “water (aqua)”, “Water/Aqua/Eau”, “purified water”… in different products. To reduce sparsity of the ingredient features and make use of ingredient information in the ingredient dictionary dataset we obtained, we try to match all ingredients to the 1750 existing ingredients. We use the [SequenceMatcher](https://docs.python.org/2/library/difflib.html) from python difflib package for this purpose. SequenceMatcher provides overall satisfactory matching results with mistakes occasionally, for example, lactic acid is mistakenly matched to acetic acid, zea mays (corn germ) oil is matched to wheat germ oil. Currently, we set a threshold of 0.25 and ingredients with match metric below that will be labelled as unknown ingredients.

#### Further Cleaning and Feature Engineering
We follow the following pipeline to cleaning our data and generate more features:
* Drop products that are not "chemical" products, like makeup brushes, cleaning devices.
* Merge some categories.
* Split 'size' column to a number and unit, do unit conversion as necessary
* Compute 'price/size'
* Ingredient features:
    * Find number of inactive and active ingredient
    * Whether the ingredients are in alphabatical order -- most companies like to list ingredient in a descending order of their quantity in the product, some companies just list ingredients alphabatically.
    * Count how many ingredients in a product have a certain rating (how many ingredient rated as Good/Average etc.)
    * Count how many ingredients in a product belongs to a certain category (how many antioxidants/sunscreen etc.)
    * Count some special categories of ingredients. For example, peptides, ingredients called "xxx extract"...
    * Compute average ingredient rating. For inactive ingredient, we also consider two types of weighted average.
    * Binary matrix indicating all ingredients' presense in each product. We may need dimensionality reduction techniques to preprocess these features for some machine learning models.
    
Datacleaning notebook can be found here: [data_cleaning.ipynb](https://github.com/NoxMoon/inside_beauty/blob/master/data_cleaning/data_cleaning.ipynb)

#### Image Preprocessing and Logo Image Filtering
We preprocessed all images to size 128 * 128. Some products on Beautypedia do not have real product photos but a logo for the brand. We build a simple classifier trained on hand picked small data set (with 104 logo samples and 283 non-logo samples) to filter these log images. We are left with 6324 unique non-logo images.

Image preprocessing and filtering notebook can be found here: [image_preprocessing.ipynb](https://github.com/NoxMoon/inside_beauty/blob/master/data_cleaning/image_preprocessing.ipynb)
[logo_image_filter.ipynb](https://github.com/NoxMoon/inside_beauty/blob/master/data_cleaning/logo_image_filter.ipynb)

## Exploratory Data Analysis

#### Visualization
We explored the following aspects with graphical EDA:
* Unique products
* Missing values
* Number of products by category
* Number of products by brand
* Price v.s. category
* Price v.s. brand
* Price v.s. ingredient
    * Is price related to the number of ingredient?
    * Is price related to the quality of ingredient?
* Price v.s. ingredient category
    * What kind of ingredient is more associated with expensive products?
    * Do those categories have higher rating?
* Ingredient Frequency

The notebook can be found here: [exploratory_data_analysis.ipynb](https://github.com/NoxMoon/inside_beauty/blob/master/eda/exploratory_data_analysis.ipynb)

#### Statistical test
In graphical EDA, we have seen many factors can contribute to cosmetic products' price. We can also use statistical test to evaluate the significance of these variables. We mainly looked at the following variables:
* Product Category (ANOVA and pairwise t-test)
* Brand (ANOVA and pairwise t-test)
* Ingredient
    * Number of ingredient (slope test)
    * Ingredient rating (slope test)
    * Ingredient category (F test)
    * Individule ingredient (chi2 test)
    
Statistical test notebook can be found here: [statistical_test_price.ipynb](https://github.com/NoxMoon/inside_beauty/blob/master/eda/statistical_test_price.ipynb)

## Machine Learning

### tSNE with ingredient features

We attempted to run some tSNE plots with ingredient features, and see if products belonging to the same category come close together in tSNE plot. It turns out it's not easy to separate different categories in a tSNE plot. The reason may be that there are too much noise with all the ingredient count. Those ingredient that are less relevant to product category contribute equally to the tSNE model, making it hard to reveal the pattern related to product category. However, after pruning individuel ingredients (by choosing ingredients with high chi2 statistics with category) and add other ingredient-related features such as count and rating, we are able to see a vague cluster of cleansers (red dots), and a cluster which is mostly a mixture of sunscreen and daytime moisturizer (purple and brown dots):

![tsne](documents/images/tsne.png)

Still, the tSNE plot looks quite noisy. Eye creams, nighttime moisturizer and serums tend to mix together, which may not be a surprise as they are similar after all. We would want to do supervised learning to see if we can distiguish product category further.

tSNE plots can be found here: [tSNE.ipynb](https://github.com/NoxMoon/inside_beauty/blob/master/machine_learning/tSNE.ipynb)

### Product category classification with ingredient features

We attempted to predict a product's category using only ingredient related features. As one product may belong to multiple categories, we can do one vs rest classification to tackle the multilabel problem. We use scikit-learn's OneVsRestClassifier for this task.

The ingredients are like the words in documents, thus many techniques for text data can be applied here. We can create a "bag of ingredients" matrix, where the matrix elements are binary indicators of whether a certain ingredient exists in a certain product. In addition, we also have "bag of ingredient categories" features that counts how many ingredients of an ingredient category are in a product. As in text data, Naive Bayes can be a good baseline model. We use Bernoullie Naive Bayes for binary features and Multinomial Naive Bayes for count features.

There are also some general ingredient features that could potentially be useful, such as number of ingredients. We can make use of these features though model stacking: the naive bayes models will serve as first layer models, then the predicted probabilities can be joined with general ingredient features to feed in the final model.
The training pipeline is as follows:

<img src="documents/images/product_category_flow_chart.png" width="800" />

We focused on 16 categories which have more than 100 products in the dataset.

#### Bernoulli Naive Bayes with binary ingredient features

The Bernoulli Naive Bayes serves as a good baseline model. It achieves 0.3724 Hamming score on training set with cross validation, and 0.3588 on test set. Naive Bayes also allows us to indentify the key ingredient associated with each category, for example:

* Acne Treatment products: benzoyl peroxide, BHA
* Cleansers: solium xxx (sodium salt of fatty acids)
* Exfoliants: AHA, BHA
* Lipsticks: Coloring Agents/Pigments
* Sunscreens: octocrylene, homosalate... (common sunscreen agents)

#### Multinomial Naive Bayes with ingredient category count features
We achieved 0.2707 Hamming score using ingredient categories alone. The results is not good enough by itself, but the predictions can be useful features for the final model.

#### Final Prediction with LightGBM

We stacked general ingredient features (number of ingredient, average/weighted rating of ingredients in products) and the predictions of the above two models to train the final model --- LightGBMClassifier. The final Hamming score is 0.4837 for cross-validation on training set and 0.5111 on test set, which is a significant improvement compare to Naive Bayes. We also improved the auc score of 10 product categories.

<img src="documents/images/auc_compare.png" width="600" /> 

We can also visualize the model predictions using confusion matrix and find the pairs of categories that our model get confused with.
![confmat](documents/images/confmat.png)

Overall, the results do agree with our life experience. The model gets confused on similar categories. For example, daytime moisturizers often have sunscreen ingredients in it, so sometimes our model cannot distiguish sunscreens and daytime moisturizer. Nighttime moisturizer, eye creams and serum are another group that our model get confused a lot in real life, they are all products that are supposed boost hydration and may have some special functions such as anti-aging, reduce hyperpigmentation... It is interesting to see that face masks got confused with cleansers, Exfoliants, and nighttime moisturizer. This is because there are typically two types of face masks: cleansing mask, which may have similar ingredient like cleansers and exfoliants. Another is the so called "sleeping mask", which you can wear overnight. They are typically like a heavy nighttime moisturizer.

Product category classification notebook can be found here: [product_category_classification.ipynb](https://github.com/NoxMoon/inside_beauty/blob/master/machine_learning/product_category_classification.ipynb)

### Price Regression
We built a LightGBM regression model to predict a product's price with both ingredient and non-ingredient features, and assess the relative importance of ingredients in determining price versus other factors such as brand and packaging.

The non-ingredient features include:
* Brand
* Product category
* Size (includes numerical size and size unit)
* Packaging (CNN model prediction with product images)

We apply target encoding on brand, product category and size unit. The packaging feature is the prediction of CNN model with product images. The price prediction with image features alone has RMSE = \$34.7 for out-of-bag prediction on training set.

Ingredient features include:
* General features such us number of inactive/active ingredient, average ingredient rating.
* Count of ingredients for each ingredient category.
* 50 selected individual ingredient binary features by chi2 test with price.
* tf-idf counts on binary ingredient matrix followed by NMF (non-negative matrix factorization), select the top 50 components as features.

Model pipeline:
<img src="documents/images/price_regression_flow_chart.png" width="700" />

The table below summarized the RMSE, MAE and explained variance when using non-ingredient features or ingredient features alone, and final prediction with all features. The ingredient features are not as powerful as non-ingredient features. The five non-ingredient features alone achieved MAE = \$10.875. Adding 171 ingredient features only improved the results slightly. The most powerful features from LightGBM's feature importance are brand and product category.

|error\features| non-ingredient features only  | ingredient features only | all features |
|-|:--|:--|:--|
|RMSE(\$) (train cv)| 22.280 |27.445 |21.393 |
|RMSE(\$) (test )|18.629 | 24.803 | 17.100|
|MAE(\$) (train cv)|12.322 | 17.130|11.692 |
|MAE(\$) (test)|10.875 | 16.298| 10.257|
|explained variance (train cv)|0.610 | 0.408 |0.640 |
|explained variance (test)| 0.683 | 0.438|0.732 |

Price regression notebook can be found here: [price_regression.ipynb](https://github.com/NoxMoon/inside_beauty/blob/master/machine_learning/price_regression.ipynb)
[packaging_to_price.ipynb](https://github.com/NoxMoon/inside_beauty/blob/master/machine_learning/packaging_to_price.ipynb)

### Conclusion
We have seen our machine learning model can more or less predict the category of a cosmetic product just using the ingredient information. It is able to associate surfactant with cleanser, coloring agents/pigment with lipsticks, AHA(alpha hydroxy acid) or BHA(beta hydroxy acid) with exfoliator, sunscreen agents with daytime moisturizers and sunscreen products... Our model also helps us identify categories that are similar ingredient-wise, such as eye creams, moisturizers and serums. Based on the machine learning results, it would make more sense for a customer to use a sleeping mask as a nighttime moisturizer, but it would not be wise to use foundation as sunscreens.

From our analysis, we do see some ingredient categories that positively correlated with price are the "good" categories according to beautypedia's expert rating, which is reassuring. But also keep in mind that things like fragrance can also make products expensive, not because it is good to the skin, but because it makes the product pleasant to use. In the price regression, we found ingredient features do have predictive power in determining price, although not as powerful as factors such as brand and product category currently. 

We do note there are several limitations in the current model. For example, currently our model is not doing well in classify toners, while in reality, people can easily tell apart a toner based on its texture. The limitation stems from the lacking of quantity information in the ingredient lists. If we know the percentage of water in a product, we can probably do better in distinguishing toners and other products. In the future, we may work on the following aspects to furthur improve the current model:
* Explore other methods for ingredient matching, such as algorithms based Levenshtein distance.
* Finding more data from other resourses can help improve the models. Currently, many brands and categories do not have enough products. 
* Improve the price prediction using product images.

Finally, I want to say that while I believe understanding the formula of cosmetic products and having a knowledge of what's inside the products can make people more rational while spending their money, and have a realistic expectation in the products, I also think it is totally fine to buy products not for need, but for fun! I hope you enjoy browsing though this work and find it interesting!
