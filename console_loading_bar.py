# This class is for displaying a loading bar in the console
class ConsoleLoadingBar():

    # Create a loading bar
    def __init__(self, num_of_iterations, prefix=''):
        self.itr_current = 0
        self.itr_end = num_of_iterations
        self.prefix = prefix
        self.show()

    # Increment the loading bar
    def increment(self):
        self.itr_current += 1
        self.show()

    # Print the loading bar
    def show(self):
        percent = ("{0:.1f}").format(100 * (self.itr_current / float(self.itr_end)))
        filledLength = int(100 * self.itr_current // self.itr_end)
        bar = '█' * filledLength + '-' * (100 - filledLength)
        print(f'\r{self.prefix} |{bar}| {percent}%', end = "\r")

        # Print New Line on Complete
        if self.itr_current == self.itr_end: 
            print()