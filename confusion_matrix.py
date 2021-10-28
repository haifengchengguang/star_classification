from model1 import baseline_model
from model_learning_rate import baseline_model_lr
from pretreatment import plot_confuse, X_test, Y_test, visual, X_train, subclass_amount

print(subclass_amount)
model=baseline_model_lr(subclass_amount=subclass_amount)
list=[1,2,7,26]
for i in list:
    weighs_path = "./save_weights_1027/myAlex_"+str(i)+".h5"
    #assert os.path.exists(img_path), "file: '{}' dose not exist.".format(weighs_path)
    model.load_weights(weighs_path)
    print("i="+str(i))
    plot_confuse(model, X_test, Y_test,i)

    ##可视化卷积层
    visual(model, X_train, 4)