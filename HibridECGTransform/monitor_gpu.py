"""
Script para monitorear el uso de GPU en tiempo real durante el entrenamiento
Ayuda a identificar cuellos de botella y optimizar el procesamiento
"""

import torch
import time
import psutil
import threading
import os
from collections import deque


class GPUMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.running = False
        self.gpu_usage_history = deque(maxlen=60)  # Últimos 60 segundos
        self.memory_usage_history = deque(maxlen=60)
        self.cpu_usage_history = deque(maxlen=60)
        
    def start_monitoring(self):
        """Inicia el monitoreo en segundo plano"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("🔍 Monitor de GPU iniciado")
        
    def stop_monitoring(self):
        """Detiene el monitoreo"""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        print("⏹️ Monitor de GPU detenido")
        
    def _monitor_loop(self):
        """Bucle principal de monitoreo"""
        while self.running:
            try:
                # Monitorear GPU
                if torch.cuda.is_available():
                    device = torch.cuda.current_device()
                    
                    # Uso de memoria GPU
                    total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
                    allocated_memory = torch.cuda.memory_allocated(device) / 1e9
                    reserved_memory = torch.cuda.memory_reserved(device) / 1e9
                    memory_usage_pct = (allocated_memory / total_memory) * 100
                    
                    self.memory_usage_history.append(memory_usage_pct)
                    
                    # Simular uso de GPU (no hay API directa en PyTorch)
                    # Usamos la actividad de memoria como proxy
                    gpu_activity = min(100, memory_usage_pct * 1.2)
                    self.gpu_usage_history.append(gpu_activity)
                
                # Monitorear CPU
                cpu_usage = psutil.cpu_percent(interval=None)
                self.cpu_usage_history.append(cpu_usage)
                
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"Error en monitoreo: {e}")
                time.sleep(self.interval)
                
    def get_current_stats(self):
        """Obtiene estadísticas actuales"""
        if not torch.cuda.is_available():
            return {"error": "CUDA no disponible"}
            
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
        allocated_memory = torch.cuda.memory_allocated(device) / 1e9
        reserved_memory = torch.cuda.memory_reserved(device) / 1e9
        free_memory = total_memory - allocated_memory
        
        return {
            "gpu_memory_total": total_memory,
            "gpu_memory_allocated": allocated_memory,
            "gpu_memory_reserved": reserved_memory,
            "gpu_memory_free": free_memory,
            "gpu_memory_usage_pct": (allocated_memory / total_memory) * 100,
            "cpu_usage_pct": psutil.cpu_percent(interval=None),
            "gpu_utilization_avg": sum(self.gpu_usage_history) / len(self.gpu_usage_history) if self.gpu_usage_history else 0
        }
        
    def print_stats(self):
        """Imprime estadísticas legibles"""
        stats = self.get_current_stats()
        
        if "error" in stats:
            print(f"❌ {stats['error']}")
            return
            
        print("\n📊 ESTADÍSTICAS DE GPU:")
        print(f"   💾 Memoria GPU: {stats['gpu_memory_allocated']:.1f}GB / {stats['gpu_memory_total']:.1f}GB ({stats['gpu_memory_usage_pct']:.1f}%)")
        print(f"   🔥 Uso CPU: {stats['cpu_usage_pct']:.1f}%")
        print(f"   ⚡ Utilización GPU promedio: {stats['gpu_utilization_avg']:.1f}%")
        
        # Detectar cuellos de botella
        self._detect_bottlenecks(stats)
        
    def _detect_bottlenecks(self, stats):
        """Detecta posibles cuellos de botella"""
        warnings = []
        
        # GPU subutilizada
        if stats['gpu_utilization_avg'] < 30:
            warnings.append("⚠️ GPU subutilizada (<30%). Posible cuello de botella en CPU o carga de datos")
            
        # Memoria GPU casi llena
        if stats['gpu_memory_usage_pct'] > 90:
            warnings.append("⚠️ Memoria GPU casi llena (>90%). Considerar reducir batch_size")
            
        # CPU muy ocupada
        if stats['cpu_usage_pct'] > 90:
            warnings.append("⚠️ CPU sobrecargada (>90%). Considerar reducir num_workers")
            
        # Memoria GPU muy baja
        if stats['gpu_memory_usage_pct'] < 20:
            warnings.append("💡 Memoria GPU infrautilizada (<20%). Podrías aumentar batch_size")
            
        for warning in warnings:
            print(f"   {warning}")
            
    def generate_report(self):
        """Genera un reporte detallado"""
        if not self.gpu_usage_history:
            print("No hay datos suficientes para generar reporte")
            return
            
        avg_gpu = sum(self.gpu_usage_history) / len(self.gpu_usage_history)
        avg_memory = sum(self.memory_usage_history) / len(self.memory_usage_history)
        avg_cpu = sum(self.cpu_usage_history) / len(self.cpu_usage_history)
        
        # Detectar períodos de baja utilización
        low_usage_periods = sum(1 for usage in self.gpu_usage_history if usage < 20)
        low_usage_pct = (low_usage_periods / len(self.gpu_usage_history)) * 100
        
        print("\n📈 REPORTE DE EFICIENCIA:")
        print(f"   🎯 Utilización GPU promedio: {avg_gpu:.1f}%")
        print(f"   💾 Uso memoria GPU promedio: {avg_memory:.1f}%")
        print(f"   🔥 Uso CPU promedio: {avg_cpu:.1f}%")
        print(f"   ⏸️ Tiempo con GPU inactiva (<20%): {low_usage_pct:.1f}%")
        
        # Recomendaciones
        print("\n💡 RECOMENDACIONES:")
        if avg_gpu < 50:
            print("   - Aumentar batch_size si hay memoria disponible")
            print("   - Verificar num_workers del DataLoader")
            print("   - Considerar usar pin_memory=True")
        if low_usage_pct > 30:
            print("   - Hay períodos largos de inactividad de GPU")
            print("   - Verificar procesamiento de datos")
        if avg_memory < 30:
            print("   - GPU tiene memoria disponible, considerar modelo más grande o batch_size mayor")


def monitor_training():
    """Función auxiliar para monitorear durante entrenamiento"""
    monitor = GPUMonitor(interval=2.0)
    monitor.start_monitoring()
    
    try:
        print("Presiona Ctrl+C para detener el monitoreo y ver el reporte final")
        while True:
            time.sleep(10)
            monitor.print_stats()
    except KeyboardInterrupt:
        print("\n🛑 Deteniendo monitoreo...")
        monitor.stop_monitoring()
        monitor.generate_report()


if __name__ == "__main__":
    print("🔍 Monitor de GPU para optimización de entrenamiento")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA no disponible. Este script requiere GPU.")
        exit()
        
    monitor_training() 