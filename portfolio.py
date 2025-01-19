from prettytable import PrettyTable

class Portfolio:
    def __init__(self, cash=1e5):
        # Stock positions and prices
        self.stock1_num = 0
        self.stock1_price_cur = 0

        self.stock2_num = 0
        self.stock2_price_cur = 0

        # Cash available for trading
        self.cash = cash
        self.value = cash
        
    def show(self):
        # Display portfolio details
        table = PrettyTable(['Stock', 'Number of holdings', 'Current price'])
        table.add_row(['Stock1', self.stock1_num, self.stock1_price_cur])
        table.add_row(['Stock2', self.stock2_num, self.stock2_price_cur])
        print(table)
        print(f"Cash: {self.cash}")
        print(f"Total value: {self.value}")
    
    def check(self):
        # Verifies if the portfolio value calculation matches
        value1 = self.stock1_num * self.stock1_price_cur
        value2 = self.stock2_num * self.stock2_price_cur
        
        if abs(value1 + value2 + self.cash - self.value) > 1:
            print("Portfolio value mismatch detected.")
            
    def updatePrice(self, stock1_price=None, stock2_price=None):
        # Updates stock prices and portfolio value accordingly
        if stock1_price:
            self.value += self.stock1_num * (stock1_price - self.stock1_price_cur)
            self.stock1_price_cur = stock1_price
        
        if stock2_price:
            self.value += self.stock2_num * (stock2_price - self.stock2_price_cur)
            self.stock2_price_cur = stock2_price
            
        self.check()

    def openPosition(self, num1=0, num2=0):
        # Open positions by buying stocks and updating cash
        if num1 > 0:
            self.stock1_num = num1
            self.cash -= num1 * self.stock1_price_cur
            print(f"Opened position for Stock1: {num1} units.")
        
        if num2 > 0:
            self.stock2_num = num2
            self.cash -= num2 * self.stock2_price_cur
            print(f"Opened position for Stock2: {num2} units.")
        
        self.check()
        
    def closePosition(self):
        # Close positions by selling stocks and updating cash
        if self.stock1_num > 0:
            self.cash += self.stock1_num * self.stock1_price_cur
            print(f"Closed position for Stock1: {self.stock1_num} units.")
            self.stock1_num = 0
            
        if self.stock2_num > 0:
            self.cash += self.stock2_num * self.stock2_price_cur
            print(f"Closed position for Stock2: {self.stock2_num} units.")
            self.stock2_num = 0
            
        self.check()

    def addPosition(self, num1=0, num2=0):
        # Add more units to existing positions and update cash
        if num1 > 0:
            self.cash -= num1 * self.stock1_price_cur
            self.stock1_num += num1
            print(f"Added position for Stock1: {num1} units.")
        
        if num2 > 0:
            self.cash -= num2 * self.stock2_price_cur
            self.stock2_num += num2
            print(f"Added position for Stock2: {num2} units.")
            
        self.check()

    def getTotalValue(self):
        # Calculate the total portfolio value
        return self.cash + (self.stock1_num * self.stock1_price_cur) + (self.stock2_num * self.stock2_price_cur)

    def getPortfolioState(self):
        # Returns a snapshot of the current portfolio state (useful for RL)
        return {
            'stock1_num': self.stock1_num,
            'stock2_num': self.stock2_num,
            'cash': self.cash,
            'value': self.getTotalValue()
        }
