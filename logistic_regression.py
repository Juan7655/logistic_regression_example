import math

def main():
	with open("log_reg_data_1var(test).csv") as file:
		data = file.readlines()

	rate = 0.001
	m_error = 0

	#Following the form ax + b in the exponential base equation
	#f(x) = 1 / (1 + exp-(ax + b))
	ma = 1
	mb = 0

	for i in range(600000):
		[ma, mb, m_error] = step(ma, mb, data, rate)
		print("a:" + str(ma) + "	b:" + str(mb))
	print("error:" + str(m_error))


def step(a, b, points, learning_rate):
	error = 0

	a_gradient = 0
	b_gradient = 0
	N = float(len(points))

	for line in points:
		text = line.split(",")
		x = float(text[0])
		y = float(text[1])

		function = 1 / (1 + math.exp(-(a * x + b)))
		gradient = function * (y - function) * (math.exp(-(a * x + b))**(1 - y))
		a_gradient -= x * gradient
		b_gradient -= gradient

	new_a = a - (learning_rate *  a_gradient)
	new_b = b - (learning_rate * b_gradient)

	return [new_a, new_b, error]

if __name__ == '__main__':
	main()