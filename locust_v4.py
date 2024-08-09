from locust import HttpUser, TaskSet, task, between, events, constant
from locust.runners import MasterRunner, WorkerRunner

# Define a global variable to count the total number of users
total_users = 0

# Define a global variable to count the total number of requests
total_requests = 0

# Define a TaskSet for user behavior
class UserBehavior(TaskSet):
    # Counter to track the number of requests per user
    requests_made = 0

    @task
    def my_task(self):
        # Check if the user has made less than 5 requests
        if self.requests_made < 5:
            # Make a request to the endpoint
            self.client.get("/your-endpoint")

            # Increment the counter
            self.requests_made += 1

            # Increment the global request count
            global total_requests
            total_requests += 1

        # Stop the user if they have made 5 requests
        if self.requests_made >= 5:
            self.interrupt()

# Define an HttpUser class
class MyUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = constant(1)

    def on_start(self):
        # Increment the global user count when a new user starts
        global total_users
        total_users += 1

    def on_stop(self):
        # Decrement the global user count when a user stops
        global total_users
        total_users -= 1

# Event listener to print total users and requests after the test
@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    if isinstance(environment.runner, (MasterRunner, WorkerRunner)):
        return  # Only print stats from master or stand-alone mode

    print(f"Total users: {total_users}")
    print(f"Total requests: {total_requests}")

# Event listener to stop the test once all users are done
@events.request.add_listener
def request_event(request_type, name, response_time, response_length, response, context, exception, start_time, **kwargs):
    if total_requests >= 50:  # Stop the test once total requests hit the limit
        environment.runner.quit()

if __name__ == "__main__":
    import os
    os.system("locust -f locustfile.py --headless -u 10 -r 10 -t 20s")

