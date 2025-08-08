import numpy as np
import pandas as pd
import keras
import ml_edu.results
import ml_edu.experiment
import plotly.express as px


dataframe = pd.read_csv(r"C:\Users\1\Downloads\chicago_taxi_train.csv")#文件链接
print(f'一共有 {len(dataframe)}','行数据')

training_df = dataframe[['TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE']]
correlations = training_df.corr(numeric_only=True)
print(correlations)

first = 'The most important feature is miles '
print(first)

def liner_model(
    settings: ml_edu.experiment.ExperimentSettings,
    metrics: list[keras.metrics.Metric],
) -> keras.Model:
  inputs = {name: keras.Input(shape=(1,), name=name) for name in settings.input_features}
  concatenated_inputs = keras.layers.Concatenate()(list(inputs.values()))
  outputs = keras.layers.Dense(units=1)(concatenated_inputs)
  model = keras.Model(inputs=inputs, outputs=outputs)
  model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=settings.learning_rate),
                loss="mean_squared_error",
                metrics=metrics)
  return model

def train_model(
    experiment_name: str,
    model: keras.Model,
    dataset: pd.DataFrame,
    label_name: str,
    settings: ml_edu.experiment.ExperimentSettings,
) -> ml_edu.experiment.Experiment:
  features = {name: dataset[name].values for name in settings.input_features}
  label = dataset[label_name].values
  history = model.fit(x=features,
                      y=label,
                      batch_size=settings.batch_size,
                      epochs=settings.number_epochs)
  return ml_edu.experiment.Experiment(
      name=experiment_name,
      settings=settings,
      model=model,
      epochs=history.epoch,
      metrics_history=pd.DataFrame(history.history),
  )
print("SUCCESS: defining linear regression functions complete.")


settings_3 = ml_edu.experiment.ExperimentSettings(
    learning_rate = 0.001,
    number_epochs = 20,
    batch_size = 50,
    input_features = ['TRIP_MILES','TRIP_MINUTES'],
)
training_df['TRIP_MINUTES'] = training_df['TRIP_SECONDS']/60
metrics = [keras.metrics.RootMeanSquaredError(name='rmse')]

model_3 = liner_model(settings_3, metrics)

experiment_3 = train_model('one_feature', model_3, training_df, 'FARE', settings_3)
ml_edu.results.plot_experiment_metrics(experiment_3, ['rmse'])
ml_edu.results.plot_model_predictions(experiment_3, training_df, 'FARE')
# 添加损失曲线可视化代码
print("\n添加损失曲线可视化:")
loss_history = experiment_3.metrics_history['loss']
fig = px.line(
    x=range(1, len(loss_history) + 1),
    y=loss_history,
    labels={'x': 'Epoch', 'y': 'Loss'},
    title='Training Loss Curve'
)
fig.update_layout(
    title_font_size=20,
    xaxis_title_font_size=16,
    yaxis_title_font_size=16,
    template='plotly_white'
)
fig.show()