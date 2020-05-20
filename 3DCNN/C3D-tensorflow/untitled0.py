import pickle
import numpy as np
from keras.models import load_model
from i3d_inception import Inception_Inflated3d

#flow_test_path = ["/DATA/crossval5_8/more_f_test7.p" for i in range(5)]
#rgb_test_path = ["/DATA/crossval5_8/more_r_test7.p" for i in range(5)]
#label_test_path = ["../crossval5_8/more_l_test7.p" for i in range(5)]
#
#rgb_data = []
#flow_data = []
#labels = []
#for t in range(5):
#        rgb_data.append(pickle.load(open(rgb_test_path[t], "rb")))
#        flow_data.append(pickle.load(open(flow_test_path[t], "rb")))
#        labels.append(pickle.load(open(label_test_path[t], "rb")))
#
#acc_list = []

for t in range(1):
        rgb_model = Inception_Inflated3d(include_top=False, weights='rgb_imagenet_and_kinetics', input_shape=(None, 224, 224, 3), classes=8)
        rgb_model.load_weights("C:/Users/Krishan/Downloads/new_rgb_0.h5")
        flow_model = Inception_Inflated3d(include_top=False, weights='flow_imagenet_and_kinetics', input_shape=(None, 224, 224, 2), classes=8)
        flow_model.load_weights("C:/Users/Krishan/Downloads/new_flow0.h5")
        
        count = 0
        y_pred = []
        y_true = []
        for i in range(len(labels[t])):
                x = np.load(rgb_data[t][i])
                x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
                rgb_logits = rgb_model.predict(x)
                x = np.load(flow_data[t][i])
                x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
                flow_logits = flow_model.predict(x)
                sample_logits = rgb_logits + flow_logits
                sample_logits = sample_logits[0]
                sample_predictions = np.exp(sample_logits) / np.sum(np.exp(sample_logits))
                conf = np.amax(sample_predictions)
                sorted_indices = np.argsort(sample_predictions)[::-1]
                pred_class = sorted_indices[0]
                true_class = np.argmax(labels[t][i])
                print(pred_class,true_class)
                y_pred.append(pred_class)
                y_true.append(true_class)

                if pred_class == true_class:
                        count = count + 1

        pickle.dump(y_pred, open("y_pred_60_above"+str(t)+".p", "wb"))
        pickle.dump(y_true, open("y_true_60_above"+str(t)+".p", "wb"))
        acc = count/len(labels[t])
        acc_list.append(acc)
        print("fold: {}, no. of samples: {}, acc: {}".format(t, len(labels[t]), acc))

#print("All folds individual accuracies:", acc_list)
print("Average accuracy:", sum(acc_list)/len(acc_list))



