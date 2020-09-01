#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def convert(image):
    return image.reshape(1, image.shape[0], image.shape[1], 1)
