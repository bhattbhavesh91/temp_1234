from locust import HttpUser, TaskSet, task, between, events, LoadTestShape
import gevent

# Define a global variable to count the total number of requests
total_requests = 0

# Define a global variable to track the user count
user_count = 0

# Define a TaskSet for user behavior
class UserBehavior(TaskSet):
    requests_made = 0

    @task
    def my_task(self):
        global total_requests

        # Make a request if under the limit
        if self.requests_made < 5:
            self.client.get("/your-endpoint")
            self.requests_made += 1
            total_requests += 1

        # Stop the user after 5 requests
        if self.requests_made >= 5:
            self.user.environment.reached_requests = True
            self.interrupt()

class MyUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 2)

    def on_start(self):
        global user_count
        user_count += 1

    def on_stop(self):
        global user_count
        user_count -= 1

def stop_test(environment):
    print(f"Total requests: {total_requests}")
    environment.process_exit_code = 0
    environment.runner.quit()

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    # Stop the test when the target request count is reached
    def check_stop():
        while True:
            if total_requests >= user_count * 5:
                stop_test(environment)
            gevent.sleep(1)

    # Spawn a greenlet to monitor the request count
    gevent.spawn(check_stop)

if __name__ == "__main__":
    import os
    os.system("locust -f locustfile.py --users 10 --spawn-rate 10 --headless -t 1m")
