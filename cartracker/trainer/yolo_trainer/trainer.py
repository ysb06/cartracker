from pytrainer import Worker


class YoloTrainer(Worker):
    def work(self) -> None:
        print(self.config)
        print("I'm Working!")
