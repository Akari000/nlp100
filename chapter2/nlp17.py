with open("../data/popular-names.txt") as f:
    lines = [line.split('\t')[0] for line in f.readlines()]

for name in sorted(set(lines)):
    print(name)

# Abigail
# Aiden
# Alexander
# Alexis
# Alice
# Amanda
# Amelia
# Amy
# Andrew
# Angela
# Anna
# Annie
# Anthony
# Ashley
# Austin
# Ava
# Barbara
# Benjamin
# Bertha
# Bessie
# Betty
# Brandon
# Brian
# Brittany
# Carol
# Carolyn
# Charles
# Charlotte
# Chloe
# Christopher
# Clara
# Crystal
# Cynthia
# Daniel
# David
# Deborah
# Debra
# Donald
# Donna
# Doris
# Dorothy
# Edward
# Elijah
# Elizabeth
# Emily
# Emma
# Ethan
# Ethel
# Evelyn
# Florence
# Frances
# Frank
# Gary
# George
# Hannah
# Harper
# Harry
# Heather
# Helen
# Henry
# Ida
# Isabella
# Jacob
# James
# Jason
# Jayden
# Jeffrey
# Jennifer
# Jessica
# Joan
# John
# Joseph
# Joshua
# Judith
# Julie
# Justin
# Karen
# Kathleen
# Kelly
# Kimberly
# Larry
# Laura
# Lauren
# Liam
# Lillian
# Linda
# Lisa
# Logan
# Lori
# Lucas
# Madison
# Margaret
# Marie
# Mark
# Mary
# Mason
# Matthew
# Megan
# Melissa
# Mia
# Michael
# Michelle
# Mildred
# Minnie
# Nancy
# Nicholas
# Nicole
# Noah
# Oliver
# Olivia
# Pamela
# Patricia
# Rachel
# Rebecca
# Richard
# Robert
# Ronald
# Ruth
# Samantha
# Sandra
# Sarah
# Scott
# Sharon
# Shirley
# Sophia
# Stephanie
# Steven
# Susan
# Tammy
# Taylor
# Thomas
# Tracy
# Tyler
# Virginia
# Walter
# William