import numpy as np
import tensorflow as tf

n_samples = 1000
batch_size = 100
num_steps = 2000

# Створюємо вхідні дані
X_data = np.random.uniform(0, 10, (n_samples, 1))
y_data = 2 * X_data + 1 + np.random.normal(0, 2, (n_samples, 1))

# Ініціалізація змінних
k = tf.Variable(tf.random.normal([1]), name='slope')
b = tf.Variable(tf.zeros([1]), name='bias')


# Навчальна модель
def linear_model(X):
    return k * X + b


# Функція втрат
def loss_fn(y_pred, y_true):
    return tf.reduce_sum(tf.square(y_pred - y_true))


# Оптимізатор
optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001)

# Тренувальний цикл
for step in range(1, num_steps + 1):
    # Вибираємо випадковий міні-батч
    indices = np.random.choice(n_samples, batch_size)
    X_batch, y_batch = X_data[indices], y_data[indices]

    # Обчислення градієнтів та оновлення параметрів
    with tf.GradientTape() as tape:
        y_pred = linear_model(X_batch)
        loss = loss_fn(y_pred, y_batch)

    # Перевірка на NaN значення
    if np.any(np.isnan(loss.numpy())):
        print(f"NaN значення на кроці {step}")
        break

    gradients = tape.gradient(loss, [k, b])
    optimizer.apply_gradients(zip(gradients, [k, b]))

    # Виведення результатів
    if step % 100 == 0:
        print(f"Крок {step}: Помилка: {loss.numpy():.4f}, k = {k.numpy()[0]:.4f}, b = {b.numpy()[0]:.4f}")
