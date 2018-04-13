# simple_portfolio_optimization
A simple portfolio optimization using the Gurobi optimization package in Python. The code is created to find a minimum risk portfolio, as well as determine the efficient frontier for the given set of stocks and their respective returns. It also finds the maximum sharpe ratio portfolio along the efficient frontier of the given stock universe. This is accomplished by using the return data, and using its average as an estimation of the future returns. The covariance matrix and standard deviation values are also used to update the model to solve along the frontier.

This can, in theory, use any combination of stocks, so long as the .csv used for the return data is formatted correctly.

![Efficient Frontier Graph Output](https://github.com/austingriffith94/simple_portfolio_optimization/blob/master/data/EfficientFrontier.png "Efficient Frontier Portfolio")

There is also another set of data provided. It has a larger universe of stocks to show how the code can be used for another given set of similarly formatted data, with no changes needed in the body of the code.
![Efficient Frontier Graph Output Second](https://github.com/austingriffith94/simple_portfolio_optimization/blob/master/different_returns/data/EfficientFrontier.png "Efficient Frontier Portfolio")
