import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam

class PPOAgent:
    def __init__(self, actor, critic, clip_epsilon=0.2, ppo_epochs=4, minibatch_size=256, gamma=0.99, gae_lambda=0.95, learning_rate=3e-4):
        self.actor = actor
        self.critic = critic
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.actor_optimizer = Adam(learning_rate=learning_rate)
        self.critic_optimizer = Adam(learning_rate=learning_rate)

    def _compute_advantages(self, rewards, values, dones):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_advantage = 0
        last_value = values[-1]
        for t in reversed(range(len(rewards) - 1)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t+1] * mask - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * mask * last_advantage
        return advantages

    @tf.function
    def _train_step(self, states, actions, advantages, log_probs_old, returns):
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            # Actor loss
            dist = tfp.distributions.Categorical(probs=self.actor(states))
            log_probs_new = dist.log_prob(actions)
            ratio = tf.exp(log_probs_new - log_probs_old)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            surrogate1 = ratio * advantages
            surrogate2 = clipped_ratio * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

            # Critic loss
            critic_loss = tf.reduce_mean(tf.square(returns - self.critic(states)))

        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        return actor_loss, critic_loss

    def train(self, states, actions, rewards, dones, values):
        advantages = self._compute_advantages(rewards, values, dones)
        returns = advantages + values[:-1]
        log_probs_old = tfp.distributions.Categorical(probs=self.actor(states[:-1])).log_prob(actions)

        dataset = tf.data.Dataset.from_tensor_slices((states[:-1], actions, advantages, log_probs_old, returns)).shuffle(buffer_size=len(states)-1)
        dataset = dataset.batch(self.minibatch_size)

        for _ in range(self.ppo_epochs):
            for batch in dataset:
                self._train_step(*batch)

    def get_action(self, state):
        dist = tfp.distributions.Categorical(probs=self.actor(np.expand_dims(state, axis=0)))
        action = dist.sample()
        value = self.critic(np.expand_dims(state, axis=0))
        return action.numpy()[0], value.numpy()[0][0]
