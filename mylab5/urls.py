from django.urls import path
from .views import plot_hyperplanes

urlpatterns = [
    path(' ', plot_hyperplanes, name=' '),
]
