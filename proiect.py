"""
**Echipa**: 11-E5
**Studenti**: ARDELEAN C. DAVID, CĂLIN V. ANDREI ALEXANDRU
**Tema proiect**: D5-T2 | Analiza unei imagini din punct de vedere al calității
"""


import cv2
import os
import matplotlib.pyplot as plt

def citeste_imagine(nume_fisier):
    """Functie care creeaza un obiect imagine din fisier."""
    imagine=cv2.imread(nume_fisier)
    if imagine is None:
        raise FileNotFoundError(f"Imaginea {nume_fisier} nu a putut fi incarcata.")
    return imagine

def detecteaza_margini_canny(imagine, prag1=100, prag2=200):
    """Functie care detecteaza margini intr-o imagine utilizand algoritmul Canny."""
    imagine_gri=cv2.cvtColor(imagine, cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(imagine_gri, prag1, prag2)
    return edges,imagine_gri

def detecteaza_contururi(imagine_margini):
    """Functie care detecteaza contururi intr-o imagine."""
    contururi, _ =cv2.findContours(imagine_margini, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contururi

def afiseaza_imagini(imagine_originala, imagine_gri, imagine_margini, imagine_cu_contururi):
    """Functie care afișeaza 4 imagini: originala, gri, margini Canny si contururi."""
    fig=plt.figure(figsize=(10, 7))

    #Conversie la RGB pentru afisare corecta
    img1=cv2.cvtColor(imagine_originala, cv2.COLOR_BGR2RGB)
    img2=imagine_gri
    img3=imagine_margini
    img4=cv2.cvtColor(imagine_cu_contururi, cv2.COLOR_BGR2RGB)

    #Afisarea celor 4 imagini
    plt.subplot(2, 2, 1)
    plt.imshow(img1)
    plt.axis('off')
    plt.title("Imagine Originala")

    plt.subplot(2, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.axis('off')
    plt.title("Imagine Gri")

    plt.subplot(2, 2, 3)
    plt.imshow(img3, cmap='gray')
    plt.axis('off')
    plt.title("Margini Canny")

    plt.subplot(2, 2, 4)
    plt.imshow(img4)
    plt.axis('off')
    plt.title("Contururi Detectate")

    plt.tight_layout()
    plt.show()

def salveaza_imagine(out_img, nume_original):
    """Functie care salveaza imaginea procesata in fisier."""
    if not os.path.exists("out"):
        os.makedirs("out")
    nume_fisier=f"out/contur_{nume_original}"
    cv2.imwrite(nume_fisier, out_img)
    print(f"Imaginea procesata a fost salvata în: {nume_fisier}")

'''def main():
    """Functia main."""
    nume_fisier="yamal.jpg"  # asigură-te că există în folder
    imagine=citeste_imagine(nume_fisier)
    margini, imagine_gri=detecteaza_margini_canny(imagine)
    contururi=detecteaza_contururi(margini)

    imagine_cu_contururi=imagine.copy()
    cv2.drawContours(imagine_cu_contururi, contururi, -1, (0, 255, 0), 2)

    afiseaza_imagini(imagine, imagine_gri, margini, imagine_cu_contururi)
    salveaza_imagine(imagine_cu_contururi, nume_fisier)
'''
def main():
    """Functia main extinsa pentru procesarea mai multor imagini."""
    folder_imagini ="imagini"
    lista_fisiere=[f for f in os.listdir(folder_imagini) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if not lista_fisiere:
        print("Nu s-au gasit imagini in folderul specificat.")
        return

    for nume_fisier in lista_fisiere:
        print(f"Se proceseaza: {nume_fisier}")
        cale_completa=os.path.join(folder_imagini, nume_fisier)

        try:
            imagine=citeste_imagine(cale_completa)
            margini, imagine_gri=detecteaza_margini_canny(imagine)
            contururi=detecteaza_contururi(margini)

            imagine_cu_contururi=imagine.copy()
            cv2.drawContours(imagine_cu_contururi, contururi, -1, (0, 255, 0), 2)

            afiseaza_imagini(imagine, imagine_gri, margini, imagine_cu_contururi)
            salveaza_imagine(imagine_cu_contururi, nume_fisier)

        except FileNotFoundError as e:
            print(f"Eroare: {e}")

if __name__== "__main__":
    main()