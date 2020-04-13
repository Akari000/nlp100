import pandas as pd

filename = "../data/popular-names.txt"
names = pd.read_csv(
    filename,
    names=("col1", "col2", "col3", "col4"),
    sep="\t",
    lineterminator="\n")

names = names.groupby("col1")
names = names.col1.count()
names = names.sort_values(ascending=False)

for row in names.index:
    print(row, end="\n")

# James
# William
# Robert
# John
# Mary
# Charles
# Michael
# Elizabeth
# Joseph
# Margaret
# George
# Thomas
# David
# Richard
# Helen
# Frank
# Christopher
# Anna
# Edward
# Ruth
# Patricia
# Matthew
# Dorothy
# Emma
# Barbara
# Daniel
# Joshua
# Emily
# Linda
# Jennifer
# Sarah
# Jacob
# Jessica
# Betty
# Susan
# Mildred
# Henry
# Ashley
# Nancy
# Andrew
# Amanda
# Florence
# Marie
# Donald
# Samantha
# Melissa
# Olivia
# Madison
# Karen
# Lisa
# Stephanie
# Abigail
# Ethel
# Sandra
# Mark
# Ethan
# Carol
# Heather
# Michelle
# Isabella
# Frances
# Angela
# Kimberly
# Ava
# Shirley
# Amy
# Nicole
# Jason
# Brian
# Sophia
# Virginia
# Deborah
# Hannah
# Donna
# Minnie
# Bertha
# Cynthia
# Brittany
# Doris
# Alice
# Nicholas
# Ronald
# Mia
# Noah
# Joan
# Debra
# Tyler
# Judith
# Ida
# Alexander
# Alexis
# Mason
# Taylor
# Clara
# Liam
# Brandon
# Tammy
# Steven
# Sharon
# Harry
# Anthony
# Charlotte
# Annie
# Gary
# Jayden
# Jeffrey
# Austin
# Chloe
# Kathleen
# Justin
# Lillian
# Benjamin
# Harper
# Aiden
# Megan
# Evelyn
# Elijah
# Amelia
# Oliver
# Rebecca
# Logan
# Larry
# Bessie
# Lauren
# Lori
# Rachel
# Pamela
# Lucas
# Walter
# Julie
# Laura
# Carolyn
# Tracy
# Scott
# Kelly
# Crystal
