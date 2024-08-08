from locust import HttpUser, TaskSet, task, between, events
import gevent

# Global variables to track the total number of requests and users
total_requests = 0
total_users = 0

# TaskSet defining the user behavior
class UserBehavior(TaskSet):
    requests_made = 0

    @task
    def my_task(self):
        global total_requests

        # Ensure the user makes only 5 requests
        if self.requests_made < 5:
            self.client.get("/your-endpoint")  # Replace with your actual endpoint
            self.requests_made += 1
            total_requests += 1

        # Stop the user once they reach 5 requests
        if self.requests_made >= 5:
            self.user.environment.reached_requests = True
            self.interrupt()

class MyUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 2)

    def on_start(self):
        global total_users
        total_users += 1

    def on_stop(self):
        global total_users
        total_users -= 1

def stop_test(environment):
    print(f"Total users: {total_users}")
    print(f"Total requests: {total_requests}")
    environment.process_exit_code = 0
    environment.runner.quit()

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    def check_stop():
        while True:
            if total_requests >= total_users * 5:
                stop_test(environment)
            gevent.sleep(1)

    # Start monitoring the request count in a separate greenlet
    gevent.spawn(check_stop)

if __name__ == "__main__":
    import os
    os.system("locust -f locustfile.py --users 10 --spawn-rate 10 --headless -t 1m")
