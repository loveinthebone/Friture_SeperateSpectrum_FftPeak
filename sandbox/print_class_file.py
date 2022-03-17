# class print_class:
#     def print_sum(self, result_from_cal_class):
#        print(result_from_cal_class)

class print_class:
    def __init__(self, calculator):
        self.calculator = calculator
    def print_sum(self):
       print(self.calculator.result)