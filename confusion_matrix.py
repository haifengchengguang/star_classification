from model1 import baseline_model
from pretreatment import plot_confuse, X_test, Y_test, visual, X_train, subclass_amount

print(subclass_amount)
model=baseline_model(subclass_amount=subclass_amount)
weighs_path = "./save_weights_extinc_normal/myAlex_10.h5"
#assert os.path.exists(img_path), "file: '{}' dose not exist.".format(weighs_path)
model.load_weights(weighs_path)
plot_confuse(model, X_test, Y_test)

# 可视化卷积层
visual(model, X_train, 1)