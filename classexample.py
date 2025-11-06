class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def greet(self):
        return f"Hello {self.name}, how are you?"

#Create Object
person1 = Person("Alice", 30)
print(person1.greet())