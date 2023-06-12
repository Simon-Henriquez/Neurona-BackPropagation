from typing import List
from random import uniform
from time import sleep

from queue import Queue

class Neurona:
    """Simple Neurona"""
    def __init__(self, nombre: str = "", rango_aleatorio: tuple[int] = (-1, 1), cantidad_entradas: int = 2,
                 cte_aprendizaje: float = 1/10):
        self.nombre = nombre
        self.cantidad_entradas = cantidad_entradas
        self.estado = "Aprendizaje no terminado"
        self.bias = round(uniform(rango_aleatorio[0], rango_aleatorio[1]), 3)
        self.pesos = [round(uniform(rango_aleatorio[0], rango_aleatorio[1]), 3) for _ in range(cantidad_entradas)]
        self.cte_aprendizaje = cte_aprendizaje

    def suma_ponderada(self, *entradas: int) -> float:
        """Retorna la suma de cada entrada multiplicada por su peso.\n
        Entrada del bias considerada cte = 1"""
        z = 0
        for i, val in enumerate(entradas):
            z += self.pesos[i] * val
        return round(z + self.bias, 3)

    def activacion(self, z: float) -> int:
        """Evalúa el valor de la suma ponderada en una función de activación 'escalon'.
        Activación dinámica dependiendo del valor del bias."""
        if z >= -self.bias:
            return 1
        return 0

    def calcular_error(self, valor_deseado: int, valor_obtenido: int) -> int:
        """Resta entre el valor deseado con el valor obtenido."""
        return valor_deseado - valor_obtenido

    def actualizar_pesos(self, *entradas: int, error: float) -> None:
        """Actualiza los pesos y el bias cuando el error es distinto de cero."""
        if error == 0:
            return 0
        for i, val in enumerate(entradas):
            self.pesos[i] = round(self.pesos[i] + (self.cte_aprendizaje * error * val), 3)
        self.bias = round(self.cte_aprendizaje * error, 3)
        return 1

    def resultado(self, entradas: List[int]) -> List[int]:
        """Retorna los resultados para cierta entrada de valores usando el bias y pesos actuales."""
        result = []
        for i in range(0, len(entradas), 2):
            x1 = entradas[i]
            x2 = entradas[i+1]
            z = self.suma_ponderada(x1, x2)
            a = self.activacion(z)
            result.append(a)
            print(f"Resultado para [{x1}, {x2}]: {a}")
        return result

class BackPropagation:
    """Encargado de generar epocas hasta optimizar los pesos."""
    @staticmethod
    def descenso_gradiente(neurona: Neurona, entradas: List[int]) -> int:
        """Itera sobre la lista de entradas con sus respectivos\n
        valores esperados, indefinidamente hasta optimizar pesos."""
        cont = 0
        for i in range(0, len(entradas), neurona.cantidad_entradas+1):
            x1 = entradas[i]
            x2 =  entradas[i+1]
            z = neurona.suma_ponderada(x1, x2)
            a = neurona.activacion(z)
            y = entradas[i+2]
            e = neurona.calcular_error(y, a)
            if cont == 0:
                cont = neurona.actualizar_pesos(x1, x2, error=e)
        return cont

    @staticmethod
    def optimizar(neurona: Neurona, entradas: List[int], queue: Queue = False) -> None:
        """Calcula nuevos pesos hasta obtener los optimos.\n
        La queue es para fines visuales con la interfaz"""
        exitoso = 1
        while exitoso != 0:
            exitoso = BackPropagation.descenso_gradiente(neurona, entradas)
            sleep(0.2)
            if queue:
                queue.put((neurona.pesos[0], neurona.pesos[1], neurona.bias))
            else:
                print(f"W1: {neurona.pesos[0]}  W2: {neurona.pesos[1]}  B: {neurona.bias}")
        neurona.estado = "Aprendizaje exitoso"
        if queue:
            queue.put(None)
        print("Optimizacion finalizada")








import tkinter as tk
from tkinter import ttk
from threading import Thread
################################################################################################
# De aqui hacia abajo es puro codigo de interfaz (ignorar)
################################################################################################
class App(tk.Tk):
    def __init__(self, title: str, size: str, neurona: Neurona):
        super().__init__()
        self.title(title)
        self.geometry(f"{size[0]}x{size[1]}")
        self.resizable(0,0)
        self.neurona = neurona
        self.upper_menu = UpperMenu(self, self.neurona)
        self.bot_menu = BotMenu(self)
        self.upper_menu.left_menu = self.bot_menu
        self.mainloop()

class UpperMenu(ttk.Frame):
    def __init__(self, parent, neurona: Neurona):
        super().__init__(parent)
        self.neurona = neurona
        self.my_font = ("SimSun", 14)
        self.pack()
        self.left_menu = None
        self.create_widget()

    def create_widget(self):
        self.btn_and = tk.Button(self, text="Entrenar AND", font=self.my_font, command=self.entrenar_and)
        self.btn_or = tk.Button(self, text="Entrenar OR", font=self.my_font, command=self.entrenar_or)

        self.btn_and.grid(row=0, column=0, padx=(0,100), pady=(15,15))
        self.btn_or.grid(row=0, column=1, pady=(15,15))

    def entrenar_and(self):
        self.neurona.nombre = "AND"
        _and = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1]
        self.comenzar_entrenamiento(_and)

    def entrenar_or(self):
        self.neurona.nombre = "OR"
        _or = [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1]
        self.comenzar_entrenamiento(_or)

    def comenzar_entrenamiento(self, entradas):
        self.left_menu.imagen_title.configure(text=f"Neurona {self.neurona.nombre} Entrenando")
        self.btn_and.configure(state="disabled")
        self.btn_or.configure(state="disabled")
        self.left_menu.insert_to_table(entradas)
        q = Queue()
        t1 = Thread(target=BackPropagation.optimizar, args=(self.neurona, entradas, q))
        t2 = Thread(target=self.actualizar, args=(q,))
        t1.start()
        t2.start()

    def actualizar(self, q):
        while True:
            datos = q.get()
            if datos == None:
                break
            self.left_menu.b.configure(text=datos[2])
            self.left_menu.w_1.configure(text=datos[0])
            self.left_menu.w_2.configure(text=datos[1])
        print("t2 finished")
        self.left_menu.imagen_title.configure(text=f"Neurona {self.neurona.nombre} Entrenada")
        self.btn_and.configure(state="normal")
        self.btn_or.configure(state="normal")


class BotMenu(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.my_font = ("SimSun", 20)
        self.pack()
        self.create_widget()
    
    def create_widget(self):
        self.sub_menu_top = ttk.Frame(self)
        self.sub_menu_mid = ttk.Frame(self)
        self.sub_menu_bot = ttk.Frame(self)
        self.imagen = tk.PhotoImage(file="Neurona2.png")
        self.imagen_label = ttk.Label(self.sub_menu_top, image=self.imagen)
        self.imagen_title = ttk.Label(self.sub_menu_top, text="Neurona", font=self.my_font)
        self.b_label = ttk.Label(self.sub_menu_mid, text="B:", font=self.my_font)
        self.b = ttk.Label(self.sub_menu_mid, text="0", font=self.my_font)
        self.w_1_label = ttk.Label(self.sub_menu_mid, text="W1:", font=self.my_font)
        self.w_1 = ttk.Label(self.sub_menu_mid, text="0", font=self.my_font)
        self.w_2_label = ttk.Label(self.sub_menu_mid, text="W2:", font=self.my_font)
        self.w_2 = ttk.Label(self.sub_menu_mid, text="0", font=self.my_font)
        self.table = ttk.Treeview(self.sub_menu_bot, height=4, columns=("X1", "X2", "Salida"), show="headings")
        self.table.heading("X1", text="X1")
        self.table.heading("X2", text="X2")
        self.table.heading("Salida", text="Salida")
        self.table.column("X1", anchor="center", width=130)
        self.table.column("X2", anchor="center", width=130)
        self.table.column("Salida", anchor="center", width=130)
        self.sub_menu_top.pack(pady=(20,60))
        self.sub_menu_mid.pack(pady=(0,40))
        self.sub_menu_bot.pack()
        self.imagen_title.pack()
        self.imagen_label.pack()
        self.b_label.grid(row=0, column=0, padx=(0,2), pady=10)
        self.b.grid(row=0, column=1, padx=(0,50), pady=10)
        self.w_1_label.grid(row=0, column=2, padx=(0,2), pady=10)
        self.w_1.grid(row=0, column=3, padx=(0,50), pady=10)
        self.w_2_label.grid(row=0, column=4, padx=(0,2), pady=10)
        self.w_2.grid(row=0, column=5, padx=(0,0), pady=10)
        self.table.pack()
        style = ttk.Style()
        style.configure("Treeview", font=("SimSun", 20), rowheight=40)
        style.configure("Treeview.Heading", font=("SimSun", 20))
        self.insert_to_table([0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1])

    def insert_to_table(self, entradas: List[int]):
        for i in self.table.get_children():
            self.table.delete(i)
        for i in range(0, len(entradas), 3):
            self.table.insert(parent="", index=i, values=(entradas[i], entradas[i+1], entradas[i+2]))

def main():
    # neurona_and = Neurona(cte_aprendizaje=1/10)
    # datos_and = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1]
    # print(neurona_and)
    # BackPropagation.optimizar(neurona_and, datos_and)
    # print(neurona_and)
    # sleep(2)
    # prueba_and = [0, 0, 0, 1, 1, 0, 1, 1]
    # neurona_and.resultado(prueba_and)
    # print("\n")
    # neurona_or = Neurona(cte_aprendizaje=1/10)
    # datos_or = [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1]
    # print(neurona_and)
    # BackPropagation.optimizar(neurona_or, datos_or)
    # print(neurona_or)
    # sleep(4)
    # prueba_or = [0, 0, 0, 1, 1, 0, 1, 1]
    # neurona_or.resultado(prueba_or)


    neurona = Neurona(rango_aleatorio=(-2, 2), cte_aprendizaje=1/100)
    App("Simple Neurona de Dos Entradas", (1080,720), neurona)

if __name__ == "__main__":
    main()