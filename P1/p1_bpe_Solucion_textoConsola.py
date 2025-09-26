from typing import Dict, Iterable, List, Optional, Tuple


class ByteLevelBPE:
    """
    Implementación básica de BPE a nivel de bytes.
    - Los tokens iniciales son bytes individuales (0..255).
    - Durante el entrenamiento se obtienen los pares de tokens adyacentes más frecuentes y se fusionan, todo ello de forma iterativa.
    - La codificación (`encode`) aplica las fusiones aprendidas en orden.
    """

    def __init__(self):
        self.merges: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
        self.vocab: Dict[Tuple[int, ...], int] = {}
        self.id2bytes: List[Tuple[int, ...]] = []

    @staticmethod
    def _to_byte_tokens(s: str) -> List[Tuple[int, ...]]:
        """
        Devuelve una lista de tokens como tuplas de bytes individuales
        """
        b = s.encode("utf-8")
        return [(x,) for x in b]

    @staticmethod
    def _count_pairs(lines_tokens: List[List[Tuple[int, ...]]]) -> Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], int]:
        """
        Obtiene las frecuencias de pares de tokens adyacentes en todas las líneas
        """
        pairs = {}
        for tokens in lines_tokens:
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pairs[pair] = pairs.get(pair, 0) + 1
        return pairs

    @staticmethod
    def _merge_in_line(line: List[Tuple[int, ...]],
                       pair: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> List[Tuple[int, ...]]:
        """
        Fusiona todas ocurrencias del par `pair` en una línea (sin solapamiento)
        """
        if len(line) < 2:
            return line
            
        new_line = []
        i = 0
        while i < len(line):
            # Si encontramos el par y no estamos al final
            if i < len(line) - 1 and (line[i], line[i + 1]) == pair:
                # Fusionar los tokens
                merged_token = line[i] + line[i + 1]
                new_line.append(merged_token)
                i += 2  # Saltar ambos tokens del par
            else:
                new_line.append(line[i])
                i += 1
        return new_line

    def train(self, lines: Iterable[str], vocab_size: int = 1000, max_merges: Optional[int] = None):
        """
        Aprende las fusiones del BPE y construye los vocabularios.
        """
        # Convertir todas las líneas a tokens de bytes
        lines_tokens = [self._to_byte_tokens(line) for line in lines]
        
        # Inicializar vocabulario con todos los bytes posibles (0-255)
        self.vocab = {(i,): i for i in range(256)}
        self.id2bytes = [(i,) for i in range(256)]
        
        # Limitar el número de fusiones
        if max_merges is None:
            max_merges = vocab_size - 256
        
        current_vocab_size = 256
        merge_count = 0
        
        while current_vocab_size < vocab_size and merge_count < max_merges:
            # Contar pares de tokens adyacentes
            pairs = self._count_pairs(lines_tokens)
            
            if not pairs:
                break
                
            # Encontrar el par más frecuente
            best_pair = max(pairs, key=pairs.get)
            
            # Agregar la fusión a la lista
            self.merges.append(best_pair)
            
            # Crear el nuevo token fusionado
            new_token = best_pair[0] + best_pair[1]
            new_id = current_vocab_size
            
            # Actualizar vocabulario
            self.vocab[new_token] = new_id
            self.id2bytes.append(new_token)
            
            # Aplicar la fusión a todas las líneas
            lines_tokens = [self._merge_in_line(line, best_pair) for line in lines_tokens]
            
            current_vocab_size += 1
            merge_count += 1

    def encode(self, text: str) -> List[int]:
        """
        Convierte el texto de entrada en una lista de token IDs.
        """
        # Convertir texto a tokens de bytes
        tokens = self._to_byte_tokens(text)
        
        # Aplicar todas las fusiones aprendidas en orden
        for merge_pair in self.merges:
            tokens = self._merge_in_line(tokens, merge_pair)
        
        # Convertir tokens a IDs
        result = []
        for token in tokens:
            if token in self.vocab:
                result.append(self.vocab[token])
            else:
                # Si el token no está en el vocabulario, usar tokens de bytes individuales
                for byte_val in token:
                    result.append(byte_val)
        
        return result

    def decode(self, ids: List[int]) -> str:
        """
        Convierte una lista de token IDs en texto.
        """
        # Convertir IDs a bytes
        byte_sequence = []
        for token_id in ids:
            if token_id < len(self.id2bytes):
                byte_sequence.extend(self.id2bytes[token_id])
            else:
                # Si el ID no es válido, agregar como byte individual
                byte_sequence.append(token_id)
        
        # Convertir bytes a string
        try:
            return bytes(byte_sequence).decode("utf-8")
        except UnicodeDecodeError:
            # En caso de error de decodificación, usar reemplazo
            return bytes(byte_sequence).decode("utf-8", errors="replace") 
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokeniza un texto.
        """
        # Convertir texto a tokens de bytes
        tokens = self._to_byte_tokens(text)
        
        # Aplicar todas las fusiones aprendidas en orden
        for merge_pair in self.merges:
            tokens = self._merge_in_line(tokens, merge_pair)
        
        # Convertir tokens a strings legibles
        result = []
        for token in tokens:
            try:
                # Intentar decodificar como UTF-8
                token_str = bytes(token).decode("utf-8")
                result.append(token_str)
            except UnicodeDecodeError:
                # Si no se puede decodificar, mostrar como bytes
                result.append(f"<{','.join(map(str, token))}>")
        
        return result 


if __name__ == "__main__":
    import sys
    import pickle
    
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python p1_bpe.py train <input_train_corpus> <output_model_file> [vocab_size]")
        print("  python p1_bpe.py eval <input_model_file> <input_text>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "train":
        if len(sys.argv) < 4:
            print("Error: Se requieren al menos 3 argumentos para entrenar")
            sys.exit(1)
            
        input_file = sys.argv[2]
        output_file = sys.argv[3]
        vocab_size = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
        
        # Entrenar el modelo
        bpe = ByteLevelBPE()
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            print(f"Entrenando BPE con {len(lines)} líneas, vocab_size={vocab_size}")
            bpe.train(lines, vocab_size=vocab_size)
            
            # Guardar el modelo
            with open(output_file, 'wb') as f:
                pickle.dump(bpe, f)
                
            print(f"Modelo guardado en {output_file}")
            print(f"Vocabulario final: {len(bpe.vocab)} tokens")
            
        except FileNotFoundError:
            print(f"Error: No se pudo encontrar el archivo {input_file}")
            sys.exit(1)
        except Exception as e:
            print(f"Error durante el entrenamiento: {e}")
            sys.exit(1)
    
    elif command == "eval":
        if len(sys.argv) < 4:
            print("Error: Se requieren al menos 3 argumentos para evaluar")
            sys.exit(1)
            
        model_file = sys.argv[2]
        input_text = sys.argv[3]
        
        try:
            # Cargar el modelo
            with open(model_file, 'rb') as f:
                bpe = pickle.load(f)
            
            # Evaluar
            print(f"Texto original: {input_text}")
            
            # Tokenizar
            tokens = bpe.tokenize(input_text)
            print(f"Tokens: {tokens}")
            
            # Codificar
            encoded = bpe.encode(input_text)
            print(f"IDs codificados: {encoded}")
            
            # Decodificar
            decoded = bpe.decode(encoded)
            print(f"Texto decodificado: {decoded}")
            
            # Verificar que la codificación/decodificación es correcta
            if input_text == decoded:
                print("Codificación/decodificación correcta")
            else:
                print("Error en codificación/decodificación")
            
        except FileNotFoundError:
            print(f"Error: No se pudo encontrar el archivo {model_file}")
            sys.exit(1)
        except Exception as e:
            print(f"Error durante la evaluación: {e}")
            sys.exit(1)
    
    else:
        print(f"Comando desconocido: {command}")
        print("Comandos válidos: train, eval")
        sys.exit(1)
