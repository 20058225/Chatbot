from ml.priority import classify_priority

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_priority_high():
    assert classify_priority("Urgent: server down!") == "high"

def test_priority_medium():
    assert classify_priority("The printer is acting weird") == "medium"

def test_priority_low():
    assert classify_priority("Can you explain logging?") == "low"

def test_priority_empty():
    assert classify_priority("") == "low"

def test_priority_none():
    assert classify_priority(None) == "low" 