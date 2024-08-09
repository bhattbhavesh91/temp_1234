from locust import HttpUser, TaskSet, task, between, events
from locust.runners import MasterRunner, WorkerRunner
import gevent
from gevent.lock import BoundedSemaphore

# Define global variables
total_users = 0
total_requests = 0
max_users = 10
user_semaphore = BoundedSemaphore(max_users)

class UserBehavior(TaskSet):
    def on_start(self):
        global total_users
        if not user_semaphore.acquire(blocking=False):
            self.interrupt()
            return
        total_users += 1

    @task
    def my_task(self):
        if self.user.requests_made < 5:
            self.client.get("/your-endpoint")
            self.user.requests_made += 1
            global total_requests
            total_requests += 1
        else:
            self.interrupt()

    def on_stop(self):
        user_semaphore.release()

class MyUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 2)
    requests_made = 0

@events.init.add_listener
def on_locust_init(environment, **kwargs):
    if isinstance(environment.runner, MasterRunner):
        gevent.spawn(check_all_users_done, environment)

def check_all_users_done(environment):
    while True:
        if total_requests >= max_users * 5:
            environment.runner.quit()
            return
        gevent.sleep(1)

@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    if not isinstance(environment.runner, WorkerRunner):
        print(f"Total users: {total_users}")
        print(f"Total requests: {total_requests}")

if __name__ == "__main__":
    import os
    os.system("locust -f locustfile.py")