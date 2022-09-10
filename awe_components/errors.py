#!/usr/bin/env python3
# Copyright 2022, Educational Testing Service

class AWE_Workbench_Error(Exception):
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text

class LexiconMissingError(AWE_Workbench_Error):
    pass
