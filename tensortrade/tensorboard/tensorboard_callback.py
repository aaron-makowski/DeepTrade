import tensorflow as tf
import numpy as np
from tensorflow.python.keras.callbacks import TensorBoard


class TensorBoardCallback(TensorBoard):
    def on_epoch_end(self, episode: int, logs=None):
        metrics = logs.copy()

        writer = self._get_writer(self._train_run_name)
        with writer.as_default():
            # process known metrics first, while removing them from dict
            with tf.name_scope('Model Performance'):
                tf.summary.scalar('loss (prediction error)', metrics.pop('loss'), step=episode)
                tf.summary.scalar('total_reward', metrics.pop('total_reward'), step=episode)

            with tf.name_scope('Portfolio'):
                tf.summary.scalar('profit_loss', metrics.pop('profit_loss'), step=episode)
                tf.summary.scalar('net_worth', metrics.pop('net_worth'), step=episode)

            with tf.name_scope('Metrics'):
                tf.summary.scalar('max_drawdown', metrics.pop('max_drawdown'), step=episode)
                tf.summary.scalar('avg_drawdown', metrics.pop('avg_drawdown'), step=episode)
                tf.summary.scalar('sharpe_ratio', metrics.pop('sharpe_ratio'), step=episode)

            with tf.name_scope('Trades'):
                tf.summary.scalar('trades_num', metrics.pop('trades_num'), step=episode)
                tf.summary.scalar('traded_amount', metrics.pop('traded_amount'), step=episode)
                tf.summary.scalar('commissions', metrics.pop('commissions'), step=episode)

            with tf.name_scope('General'):
                tf.summary.scalar('epsilon (randomness)', metrics.pop('epsilon'), step=episode)

            # now process other metrics
            for metric, value in metrics.items():
                if np.isscalar(value):
                    tf.summary.scalar(metric, value, step=episode)
                else:
                    tf.summary.histogram(metric, value, step=episode)

        super(TensorBoardCallback, self).on_epoch_end(episode)
