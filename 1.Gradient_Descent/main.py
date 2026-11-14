# Gradient descent - pure python implementation
# No ML libraries only NATIVE MATH!!

import math
def quadratic_function(x,a,b,c):
    '''Quadratic funtions f(x) = ax² + bx + c '''
    return a * x * x + b * x + c

def quadratic_gradient(x,a,b):
    '''Gradient(derivative) of quadratic funtions ∇f(x) = 2ax + b'''
    return 2 * a * x + b

def gradient_descent(initial_x, lr, threshold, max_iter, a=1, b=0, c=0):
    '''Gradient descent optimization algorithm'''
    x = initial_x
    history = []

    for i in range(max_iter):
        grad = quadratic_gradient(x,a,b)
        fx = quadratic_function(x,a,b,c)

        history.append({'iteration': i + 1, 'x': x, 'fx': fx, 'gradient': grad})

        # CONVERGENCE condition = gradient is small enough
        if abs(grad) < threshold:
            return {
                'converged': True,
                'iterations': i + 1,
                'final_x': x,
                'final_fx': quadratic_function(x,a,b,c),
                'history': history
            }

        # Gradient descent update: x = x - lr * gradient
        x = x - lr * grad              #lr = learning rate

    # Reached maximum iterations
    return {
        'converged': False,
        'iterations': max_iter,
        'final_x': x,
        'final_fx': quadratic_function(x,a,b,c),
        'history': history,
        'error': 'Reached maximum iterations'
    }    


# Run test
if __name__ == '__main__':
    result = gradient_descent(0,0.1,1e-6,1000,1,-6,11)
    print(f"Test result: {result}")
    print(f"Final position: x = {result['final_x']:.6f}")
    print(f"Final loss: f(x) = {result['final_fx']:.6f}")
    print(f"Iterations: {result['iterations']}")
    print(f"Converged: {result['converged']}")
