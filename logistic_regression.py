import math
import pandas
import time
import matplotlib.pyplot as plt

iterations = 2000


def main():
	data = pandas.read_csv("log_reg_data_1var(test).csv")
	data.columns = ["x", "y"]
	rate = 0.01

	# Following the form ax + b in the exponential base equation
	# f(x) = 1 / (1 + exp-(ax + b))
	ma = 1
	mb = 0
	m_error = 0
	result = []
	start_time = time.time()

	for i in range(iterations):
		[ma, mb, m_error] = step(ma, mb, data, rate)
		result.append([ma, mb, m_error])
	res = pandas.DataFrame(data=result, columns=["a", "b", "error"])
	print("a:" + str(ma) + "	b:" + str(mb) + "	error:" + str(m_error))
	print("Execution time: %s seconds" % (time.time() - start_time))

	# x values for the model lines in the graph
	data_x_max = data["x"].max()
	data_x_min = data["x"].min()
	x_val = pandas.DataFrame(data=[data_x_min + i * (data_x_max - data_x_min) / 1000 for i in range(0, 1000)])

	# plotting final model line
	linear_data = pandas.DataFrame(data=linear_model(ma, mb, x_val))
	y_val = linear_data.applymap(lambda x: 1 / (1 + math.exp(x)))
	plt.plot(x_val, y_val, zorder=3, c='g', linewidth=2)

	# plotting model line in each iteration(guess)
	# step size optimized to reduce excessive time
	for i in range(0, len(res), max(int(iterations / 6000), 1)):
		linear_data = linear_model(res["a"][i], res["b"][i], x_val)
		y_val = linear_data.applymap(lambda x: 1 / (1 + math.exp(x)))
		plt.plot(x_val, y_val, zorder=2, c='b', linewidth=.1)

	# base scatter graph (original data points)
	plt.scatter(data["x"], data["y"], zorder=1, s=3)
	plt.xlabel("x")
	plt.ylabel("y")
	plt.show()

	# plotting error change throw each learning iteration
	x_val = [i for i in range(iterations)]
	plt.xlabel("Learning Iteration")
	plt.ylabel("Error Value")
	plt.plot(x_val, res["error"], c='r', linewidth=1)
	plt.show()


def step(a, b, points, learning_rate):
	# base functions
	linear_mod = linear_model(a, b, points["x"])
	expo_function = linear_mod.map(lambda x: 1 / (1 + math.exp(x)))
	error = 1 - (expo_function ** points["y"])*((1 - expo_function) ** (1 - points["y"]))

	# gradient computing
	gradient = expo_function*(points["y"] - expo_function)*(linear_mod.apply(math.exp) ** (1 - points["y"]))
	a_gradient = points["x"].dot(gradient)
	b_gradient = gradient.sum()

	# learning adjustment
	new_a = a + (learning_rate * a_gradient)
	new_b = b + (learning_rate * b_gradient)

	return [new_a, new_b, error.sum()]


def linear_model(a, b, data):
	return -(a * data + b)


if __name__ == '__main__':
	main()
