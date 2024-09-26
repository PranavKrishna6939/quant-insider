import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Hyperparams
n_units = 400
layers = 4
n_batch = 1024
n_epochs = 100

call_df = pd.read_csv('/home/jjbigdub/gitrepo/quant-insider/call_data.csv')
call_df = call_df.drop(columns=['datetime', 'expiry_date', 'right','open','high','low' ,'log_return'])
call_df = call_df.dropna(axis=0)

# put_df = pd.read_csv('/home/jjbigdub/gitrepo/quant-insider/put_data.csv')
# put_df = put_df.drop(columns=['datetime', 'expiry_date', 'right','open','high','low' ,'log_return'])
# put_df = put_df.dropna(axis=0)

call_X_train, call_X_test, call_y_train, call_y_test = train_test_split(call_df.drop(['close_option'], axis=1), call_df.close_option,
                                                                        test_size=0.01, random_state=42)

# put_X_train, put_X_test, put_y_train, put_y_test = train_test_split(put_df.drop(['close_option'], axis=1), put_df.close_option,
#                                                                     test_size=0.01, random_state=42)

model = keras.Sequential()
model.add(keras.layers.Dense(n_units, input_dim=call_X_train.shape[1]))
model.add(keras.layers.LeakyReLU())
for _ in range(layers - 1):
    model.add(keras.layers.Dense(n_units))
    model.add(keras.layers.LeakyReLU())
model.add(keras.layers.Dense(1, activation='relu'))

model.summary()

# Initial Training at lr = 1e-5 for 100 epochs
model.compile(loss='mse',optimizer=keras.optimizers.Adam(learning_rate=1e-5))
history = model.fit(
    call_X_train, 
    call_y_train, 
    batch_size=n_batch, 
    epochs=n_epochs, 
    validation_split=0.01, 
    callbacks=[keras.callbacks.TensorBoard()], 
    verbose=1
)
call_y_pred = model.predict(call_X_test)
print('test set mse', np.mean(np.square(call_y_test - np.reshape(call_y_pred, call_y_pred.shape[0]))))


# Training at lr = 1e-6 for 20 epochs
model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=1e-6))
history = model.fit(call_X_train, call_y_train, 
                    batch_size=n_batch, epochs=20, 
                    validation_split = 0.01,
                    callbacks=[keras.callbacks.TensorBoard()],
                    verbose=1)

call_y_pred2 = model.predict(call_X_test)
print('test set mse', np.mean(np.square(call_y_test - np.reshape(call_y_pred2, call_y_pred2.shape[0]))))


# Training at lr = 1e-7 for 10 epochs
model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=1e-7))
history = model.fit(call_X_train, call_y_train, 
                    batch_size=n_batch, epochs=10, 
                    validation_split = 0.01,
                    callbacks=[keras.callbacks.TensorBoard()],
                    verbose=1)

call_y_pred3 = model.predict(call_X_test)
print('test set mse', np.mean(np.square(call_y_test - np.reshape(call_y_pred3, call_y_pred3.shape[0]))))


# Training at lr = 1e-8 for 5 epochs
model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=1e-8))
history = model.fit(call_X_train, call_y_train, 
                    batch_size=n_batch, epochs=5, 
                    validation_split = 0.01,
                    callbacks=[keras.callbacks.TensorBoard()],
                    verbose=1)

call_y_pred4 = model.predict(call_X_test)
print('test set mse', np.mean(np.square(call_y_test - np.reshape(call_y_pred4, call_y_pred4.shape[0]))))
