from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    Input,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Add,
    ReLU,
    Activation,
    LSTM,
)
from tensorflow.keras.regularizers import l2


# CNN1
cnn1 = Sequential(
    [
        Input((187, 1)),
        # First Conv Block
        Conv1D(filters=64, kernel_size=5, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        # Second Conv Block
        Conv1D(filters=64, kernel_size=5, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        # Third Conv Block
        Conv1D(filters=32, kernel_size=3, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        # Dense layers
        Flatten(),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(5, activation="softmax"),
    ]
)


# CNN2, CNN4 Paper 2020
cnn2 = Sequential(
    [
        Input((187, 1)),
        # First Conv Block: 5×32
        Conv1D(filters=32, kernel_size=5, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        # Second Conv Block: 3×64
        Conv1D(filters=64, kernel_size=3, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        # Third Conv Block: 5×128
        Conv1D(filters=128, kernel_size=5, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        # Fourth Conv Block: 3×256
        Conv1D(filters=256, kernel_size=3, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        # Dense layers
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(5, activation="softmax"),
    ]
)


# CNN3, CNN4 Paper 2020, increased dropout compared to CNN2
cnn3 = Sequential(
    [
        Input((187, 1)),
        # First Conv Block
        Conv1D(filters=32, kernel_size=5, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        # Second Conv Block
        Conv1D(filters=64, kernel_size=3, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        # Third Conv Block
        Conv1D(filters=128, kernel_size=5, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.4),
        # Fourth Conv Block
        Conv1D(filters=256, kernel_size=3, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.4),
        # Dense layers
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(5, activation="softmax"),
    ]
)


# CNN4, CNN4 Paper 2020, changed dropout compared to CNN2 and CNN3
cnn4 = Sequential(
    [
        Input((187, 1)),
        # First Conv Block
        Conv1D(filters=32, kernel_size=5, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),
        # Second Conv Block
        Conv1D(filters=64, kernel_size=3, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),
        # Third Conv Block
        Conv1D(filters=128, kernel_size=5, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        # Fourth Conv Block
        Conv1D(filters=256, kernel_size=3, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.35),
        # Dense layers
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.4),
        Dense(5, activation="softmax"),
    ]
)


# CNN5, Paper 2018
# Input
inputs = Input(shape=(187, 1))

# Initial conv layer, 32 filters
x = Conv1D(filters=32, kernel_size=5, padding="same")(inputs)

# Residual Block 1
shortcut = x
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = ReLU()(x)
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = Add()([shortcut, x])
x = ReLU()(x)
x = MaxPooling1D(pool_size=5, strides=2, padding="same")(x)

# Residual Block 2
shortcut = x
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = ReLU()(x)
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = Add()([shortcut, x])
x = ReLU()(x)
x = MaxPooling1D(pool_size=5, strides=2, padding="same")(x)

# Residual Block 3
shortcut = x
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = ReLU()(x)
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = Add()([shortcut, x])
x = ReLU()(x)
x = MaxPooling1D(pool_size=5, strides=2, padding="same")(x)

# Residual Block 4
shortcut = x
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = ReLU()(x)
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = Add()([shortcut, x])
x = ReLU()(x)
x = MaxPooling1D(pool_size=5, strides=2, padding="same")(x)

# Residual Block 5
shortcut = x
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = ReLU()(x)
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = Add()([shortcut, x])
x = ReLU()(x)
x = MaxPooling1D(pool_size=5, strides=2, padding="same")(x)

# Fully connected layers
x = Flatten()(x)
x = Dense(32, activation="relu")(x)
x = Dense(32, activation="relu")(x)

# output
outputs = Dense(5, activation="softmax")(x)

# create model
cnn5 = Model(inputs=inputs, outputs=outputs)


# CNN6, Paper 2018 added dropout and batch normalization
inputs = Input(shape=(187, 1))
# Initial conv
x = Conv1D(filters=32, kernel_size=5, padding="same")(inputs)
x = BatchNormalization()(x)

# Residual Block 1
shortcut = x
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = BatchNormalization()(x)
x = Add()([shortcut, x])
x = ReLU()(x)
x = MaxPooling1D(pool_size=5, strides=2, padding="same")(x)
x = Dropout(0.2)(x)

# Residual Block 2
shortcut = x
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = BatchNormalization()(x)
x = Add()([shortcut, x])
x = ReLU()(x)
x = MaxPooling1D(pool_size=5, strides=2, padding="same")(x)
x = Dropout(0.2)(x)

# Residual Block 3
shortcut = x
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = BatchNormalization()(x)
x = Add()([shortcut, x])
x = ReLU()(x)
x = MaxPooling1D(pool_size=5, strides=2, padding="same")(x)
x = Dropout(0.2)(x)

# Residual Block 4
shortcut = x
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = BatchNormalization()(x)
x = Add()([shortcut, x])
x = ReLU()(x)
x = MaxPooling1D(pool_size=5, strides=2, padding="same")(x)
x = Dropout(0.25)(x)

# Residual Block 5
shortcut = x
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = BatchNormalization()(x)
x = Add()([shortcut, x])
x = ReLU()(x)
x = MaxPooling1D(pool_size=5, strides=2, padding="same")(x)
x = Dropout(0.25)(x)

# Fully connected layers
x = Flatten()(x)
x = Dense(32, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(32, activation="relu")(x)
x = Dropout(0.3)(x)

# output
outputs = Dense(5, activation="softmax")(x)

# Create model
cnn6 = Model(inputs=inputs, outputs=outputs)


# CNN7, Paper 2018 added batch normalization, without dropout
inputs = Input(shape=(187, 1))

# Initial conv layer
x = Conv1D(filters=32, kernel_size=5, padding="same")(inputs)
x = BatchNormalization()(x)

# Residual Block 1
shortcut = x
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = BatchNormalization()(x)
x = Add()([shortcut, x])
x = ReLU()(x)
x = MaxPooling1D(pool_size=5, strides=2, padding="same")(x)

# Residual Block 2
shortcut = x
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = BatchNormalization()(x)
x = Add()([shortcut, x])
x = ReLU()(x)
x = MaxPooling1D(pool_size=5, strides=2, padding="same")(x)

# Residual Block 3
shortcut = x
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = BatchNormalization()(x)
x = Add()([shortcut, x])
x = ReLU()(x)
x = MaxPooling1D(pool_size=5, strides=2, padding="same")(x)

# Residual Block 4
shortcut = x
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = BatchNormalization()(x)
x = Add()([shortcut, x])
x = ReLU()(x)
x = MaxPooling1D(pool_size=5, strides=2, padding="same")(x)

# Residual Block 5
shortcut = x
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = BatchNormalization()(x)
x = Add()([shortcut, x])
x = ReLU()(x)
x = MaxPooling1D(pool_size=5, strides=2, padding="same")(x)

# Fully Connected Layers
x = Flatten()(x)
x = Dense(32, activation="relu")(x)
x = Dense(32, activation="relu")(x)

# Output
outputs = Dense(5, activation="softmax")(x)

# Create model
cnn7 = Model(inputs=inputs, outputs=outputs)


# CNN8, Paper 2018 added dropout, without batch normalization
inputs = Input(shape=(187, 1))

# Initial conv layer
x = Conv1D(filters=32, kernel_size=5, padding="same")(inputs)

# Residual Block 1
shortcut = x
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = ReLU()(x)
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = Add()([shortcut, x])
x = ReLU()(x)
x = MaxPooling1D(pool_size=5, strides=2, padding="same")(x)
x = Dropout(0.1)(x)

# Residual Block 2
shortcut = x
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = ReLU()(x)
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = Add()([shortcut, x])
x = ReLU()(x)
x = MaxPooling1D(pool_size=5, strides=2, padding="same")(x)
x = Dropout(0.1)(x)

# Residual Block 3
shortcut = x
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = ReLU()(x)
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = Add()([shortcut, x])
x = ReLU()(x)
x = MaxPooling1D(pool_size=5, strides=2, padding="same")(x)
x = Dropout(0.1)(x)

# Residual Block 4
shortcut = x
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = ReLU()(x)
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = Add()([shortcut, x])
x = ReLU()(x)
x = MaxPooling1D(pool_size=5, strides=2, padding="same")(x)
x = Dropout(0.1)(x)

# Residual Block 5
shortcut = x
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = ReLU()(x)
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = Add()([shortcut, x])
x = ReLU()(x)
x = MaxPooling1D(pool_size=5, strides=2, padding="same")(x)
x = Dropout(0.1)(x)

# Fully Connected Layers
x = Flatten()(x)
x = Dense(32, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(32, activation="relu")(x)
x = Dropout(0.1)(x)

# Output
outputs = Dense(5, activation="softmax")(x)

# Create model
cnn8 = Model(inputs=inputs, outputs=outputs)


# CNN9 - changed Dropout strategy compared to CNN8
inputs = Input(shape=(187, 1))

# initial conv layer
x = Conv1D(filters=32, kernel_size=5, padding="same")(inputs)

# Residual Block 1
shortcut = x
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = ReLU()(x)
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = Add()([shortcut, x])
x = ReLU()(x)
x = MaxPooling1D(pool_size=5, strides=2, padding="same")(x)
x = Dropout(0.1)(x)

# Residual Block 2
shortcut = x
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = ReLU()(x)
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = Add()([shortcut, x])
x = ReLU()(x)
x = MaxPooling1D(pool_size=5, strides=2, padding="same")(x)
x = Dropout(0.15)(x)

# Residual Block 3
shortcut = x
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = ReLU()(x)
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = Add()([shortcut, x])
x = ReLU()(x)
x = MaxPooling1D(pool_size=5, strides=2, padding="same")(x)
x = Dropout(0.2)(x)

# Residual Block 4
shortcut = x
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = ReLU()(x)
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = Add()([shortcut, x])
x = ReLU()(x)
x = MaxPooling1D(pool_size=5, strides=2, padding="same")(x)
x = Dropout(0.25)(x)

# Residual Block 5
shortcut = x
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = ReLU()(x)
x = Conv1D(filters=32, kernel_size=5, padding="same")(x)
x = Add()([shortcut, x])
x = ReLU()(x)
x = MaxPooling1D(pool_size=5, strides=2, padding="same")(x)
x = Dropout(0.3)(x)

# Fully Connected Layers
x = Flatten()(x)
x = Dense(32, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(32, activation="relu")(x)
x = Dropout(0.3)(x)

# output
outputs = Dense(5, activation="softmax")(x)

cnn9 = Model(inputs=inputs, outputs=outputs)


# Used LSTMs
# LSTM1, 2 layers
lstm1 = Sequential()

# First LSTM layer
lstm1.add(LSTM(units=32, return_sequences=True, input_shape=(187, 1)))

# Second LSTM layer
lstm1.add(LSTM(units=32, return_sequences=False))

# Fully connected layer
lstm1.add(Dense(32, activation="relu"))

# Output layer with softmax activation for classification
lstm1.add(Dense(5, activation="softmax"))


# LSTM2, 6 layers
lstm2 = Sequential()

# First LSTM layer
lstm2.add(LSTM(units=32, return_sequences=True, input_shape=(187, 1)))

# Second LSTM layer
lstm2.add(LSTM(units=32, return_sequences=True))

# Third LSTM layer
lstm2.add(LSTM(units=32, return_sequences=True))

# Fourth LSTM layer
lstm2.add(LSTM(units=32, return_sequences=True))

# Fifth LSTM layer
lstm2.add(LSTM(units=32, return_sequences=True))

# Sixth LSTM layer
lstm2.add(LSTM(units=32, return_sequences=False))

# Fully connected layer
lstm2.add(Dense(32, activation="relu"))

# Output layer with softmax activation for classification
lstm2.add(Dense(5, activation="softmax"))


# Used DNNs

# DNN1
dnn1 = Sequential()
dnn1.add(Dense(units=64, activation="relu", input_shape=(187,)))
dnn1.add(Dense(units=32, activation="relu"))
dnn1.add(Dense(units=16, activation="relu"))
dnn1.add(Dense(units=5, activation="softmax"))


# DNN2
dnn2 = Sequential(
    [
        Dense(64, input_shape=(187,), kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.2),
        Dense(32, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.2),
        Dense(16, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.1),
        Dense(5, activation="softmax"),
    ]
)

# DNN3
dnn3 = Sequential(
    [
        Dense(64, input_shape=(187,)),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.2),
        Dense(32),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.2),
        Dense(16),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.1),
        Dense(5, activation="softmax"),
    ]
)


# DNN4
dnn4 = Sequential(
    [
        Dense(64, activation="relu", input_shape=(187,), kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(16, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.1),
        Dense(5, activation="softmax"),
    ]
)
