# MachineLearning-Clustering
ModelType : &lt;k-means, em, clarans, dbscan, affinity>

<h2>Dataset:
Housing</h2>
<br/>
https://www.kaggle.com/camnugent/california-housing-prices
<br/>

![image](https://user-images.githubusercontent.com/84762786/195782968-2e288407-a9f8-4b48-8a2e-de4d61ba6aff.png)


<h2>
Model type : &lt;k-means, em, clarans, dbscan, affinity>
</h2>
<p>
-Various data scaling methods and encoding methods
<br/>
-Various values of the model parameters for each model
<br/>
-Various values for the hyperparameters
<br/>
-Various subsets of the features of the dataset
<p>

<h3>
You can use AutoML function to automatically run different combinations of the above within a <br/>“single major function”.
</h3>

def AutoML(scaler, encoder, model,train) :
    
    if not isinstance(train, pd.DataFrame):
        raise TypeError
    # if not isinstance(scaler, dict):
    #     raise TypeError
    
    predicted = {}

    print("Encoder Type : ",encoder)
    print("Scaler Type : ",scaler)
    print("Model Type : ",model)

    hyper_param = {
    "k-means": [5],
    "em": [5],
    "clarans": {"numCluster":[4], "numLocal":[5], "maxNeighbor":[5]},
    "dbscan": {"eps":[0.1], "minSample":[4]}
}

    for e in encoder:
        e_train = Encoding(train, e)
        print("------------------------------------")
        print("Encoder Type: ",e)
        for s in scaler:
            print("------------------------------------")
            print("Scaler Type : " , s)
            s_train = Scaling(e_train,s)
            for m in model:
                print("------------------------------------")
                print("Model Type : ",m)
                predicted = predict(m, s_train)
                


AutoML(scaler, encoder, model,df_label)   

<h2>
Result :
<h2/>
    
![image](https://user-images.githubusercontent.com/84762786/195783419-b69ddbea-4b69-4ccf-88dd-96115960b78a.png)

![image](https://user-images.githubusercontent.com/84762786/195783450-9bcc84e2-79f0-45a6-8801-14d7eb3c9afa.png)

![image](https://user-images.githubusercontent.com/84762786/195783531-afb56c99-6fc0-4238-a4ae-d07b3236dd6b.png)

See more details in AutoML (5).ipynb file

<h2>User Manual</h2>
    
![image](https://user-images.githubusercontent.com/84762786/195783980-2e9f2b13-6d8e-4b7e-aaec-084bbd3c50c4.png)
