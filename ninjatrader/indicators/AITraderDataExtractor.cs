using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.IO;
using System.Linq;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.DrawingTools;
using NinjaTrader.Core.FloatingPoint;

namespace NinjaTrader.NinjaScript.Indicators
{
    public class AITraderDataExtractor : Indicator
    {
        #region Properties
        [NinjaScriptProperty]
        [Display(Name = "Export to CSV", Description = "Set to true to export data to CSV", Order = 1, GroupName = "Parameters")]
        public bool ExportToCSV { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Export Path", Description = "Path to export CSV file", Order = 2, GroupName = "Parameters")]
        public string ExportPath { get; set; }
        
        [NinjaScriptProperty]
        [Range(1, int.MaxValue)]
        [Display(Name = "Export Delay Bars", Description = "Number of bars to wait before exporting (0 to wait for last bar)", Order = 3, GroupName = "Parameters")]
        public int ExportDelayBars { get; set; }
        
        [NinjaScriptProperty]
        [Range(1, int.MaxValue)]
        [Display(Name = "Max Bars to Export", Description = "Maximum number of bars to export (from newest to oldest)", Order = 4, GroupName = "Parameters")]
        public int MaxBarsToExport { get; set; }
        
        [NinjaScriptProperty]
        [Range(10, 200)]
        [Display(Name = "Fast EMA Period", Description = "Period for Fast EMA", Order = 5, GroupName = "Indicators")]
        public int FastEMAPeriod { get; set; }
        
        [NinjaScriptProperty]
        [Range(20, 300)]
        [Display(Name = "Slow EMA Period", Description = "Period for Slow EMA", Order = 6, GroupName = "Indicators")]
        public int SlowEMAPeriod { get; set; }
        
        [NinjaScriptProperty]
        [Range(5, 50)]
        [Display(Name = "RSI Period", Description = "Period for RSI indicator", Order = 7, GroupName = "Indicators")]
        public int RSIPeriod { get; set; }
        
        [NinjaScriptProperty]
        [Range(5, 50)]
        [Display(Name = "Stochastic Period", Description = "Period for Stochastic indicator", Order = 8, GroupName = "Indicators")]
        public int StochasticPeriod { get; set; }
        
        [NinjaScriptProperty]
        [Range(10, 50)]
        [Display(Name = "Bollinger Bands Period", Description = "Period for Bollinger Bands", Order = 9, GroupName = "Indicators")]
        public int BollingerPeriod { get; set; }
        
        [NinjaScriptProperty]
        [Range(1.0, 3.0)]
        [Display(Name = "Bollinger Bands StdDev", Description = "Standard deviation for Bollinger Bands", Order = 10, GroupName = "Indicators")]
        public double BollingerStdDev { get; set; }
        
        [NinjaScriptProperty]
        [Range(5, 50)]
        [Display(Name = "MACD Fast", Description = "Fast period for MACD", Order = 11, GroupName = "Indicators")]
        public int MACDFast { get; set; }
        
        [NinjaScriptProperty]
        [Range(10, 100)]
        [Display(Name = "MACD Slow", Description = "Slow period for MACD", Order = 12, GroupName = "Indicators")]
        public int MACDSlow { get; set; }
        
        [NinjaScriptProperty]
        [Range(3, 20)]
        [Display(Name = "MACD Signal", Description = "Signal period for MACD", Order = 13, GroupName = "Indicators")]
        public int MACDSignal { get; set; }
        
        [NinjaScriptProperty]
        [Range(7, 100)]
        [Display(Name = "ATR Period", Description = "Period for Average True Range", Order = 14, GroupName = "Indicators")]
        public int ATRPeriod { get; set; }
        
        [NinjaScriptProperty]
        [Range(3, 20)]
        [Display(Name = "ADX Period", Description = "Period for ADX", Order = 15, GroupName = "Indicators")]
        public int ADXPeriod { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Debug Mode", Description = "Show detailed debugging messages", Order = 16, GroupName = "Debug")]
        public bool DebugMode { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Force Export on Realtime", Description = "Force export when reaching realtime data", Order = 17, GroupName = "Debug")]
        public bool ForceExportOnRealtime { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Button Export", Description = "Add a button to manually trigger export", Order = 18, GroupName = "Debug")]
        public bool ButtonExport { get; set; }
        #endregion
        
        #region Variables
        private EMA fastEMA;
        private EMA slowEMA;
        private RSI rsi;
        private Stochastics stoch;
        private Bollinger bbands;
        private MACD macd;
        private ATR atr;
        private ADX adx;
        private VOL volume;
        private Series<double> adxPlus;
        private Series<double> adxMinus;
        private bool hasExecutedExport = false;
        private int debugCounter = 0;
        private bool historyProcessed = false;
        private System.Windows.Controls.Button exportButton;
        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Extractor de datos para AITraderPro con análisis de patrones de reversión";
                Name = "AITraderDataExtractor";
                Calculate = Calculate.OnBarClose;
                IsOverlay = false;
                DisplayInDataBox = false;
                DrawOnPricePanel = false;
                IsSuspendedWhileInactive = true;

                // Valores por defecto
                ExportPath = @"C:\AITraderPro\data";
                MaxBarsToExport = 5000;
                ExportDelayBars = 100; // Exportar después de 100 barras
                
                // Indicadores técnicos por defecto
                FastEMAPeriod = 9;
                SlowEMAPeriod = 20;
                RSIPeriod = 14;
                StochasticPeriod = 14;
                BollingerPeriod = 20;
                BollingerStdDev = 2.0;
                MACDFast = 12;
                MACDSlow = 26;
                MACDSignal = 9;
                ATRPeriod = 14;
                ADXPeriod = 14;
                
                // Debug options
                DebugMode = true;
                ForceExportOnRealtime = true;
                ButtonExport = true;
                
                LogDebug("SetDefaults complete");
            }
            else if (State == State.Configure)
            {
                LogDebug($"Configurando indicador - ExportToCSV: {ExportToCSV}, Path: {ExportPath}");
                
                // Añadir indicadores
                try {
                    fastEMA = EMA(FastEMAPeriod);
                    LogDebug($"Fast EMA ({FastEMAPeriod}) configurado");
                    
                    slowEMA = EMA(SlowEMAPeriod);
                    LogDebug($"Slow EMA ({SlowEMAPeriod}) configurado");
                    
                    rsi = RSI(RSIPeriod, 1);
                    LogDebug($"RSI ({RSIPeriod}) configurado");
                    
                    stoch = Stochastics(StochasticPeriod, 3, 3);
                    LogDebug($"Stochastics ({StochasticPeriod}) configurado");
                    
                    // Corregido: Usar Bollinger en lugar de BollingerBands y convertir parámetros a int
                    bbands = Bollinger(Convert.ToInt32(BollingerPeriod), Convert.ToInt32(BollingerStdDev));
                    LogDebug($"Bollinger ({BollingerPeriod}, {BollingerStdDev}) configurado");
                    
                    macd = MACD(MACDFast, MACDSlow, MACDSignal);
                    LogDebug($"MACD ({MACDFast}, {MACDSlow}, {MACDSignal}) configurado");
                    
                    atr = ATR(ATRPeriod);
                    LogDebug($"ATR ({ATRPeriod}) configurado");
                    
                    adx = ADX(ADXPeriod);
                    LogDebug($"ADX ({ADXPeriod}) configurado");
                    
                    volume = VOL();
                    LogDebug("Volume configurado");
                    
                    // Crear series personalizadas para los componentes de ADX
                    adxPlus = new Series<double>(this);
                    adxMinus = new Series<double>(this);
                    LogDebug("Series ADX Plus/Minus configuradas");
                }
                catch (Exception ex) {
                    Print($"ERROR en configuración: {ex.Message}");
                    if (ex.InnerException != null) {
                        Print($"  Inner Exception: {ex.InnerException.Message}");
                    }
                    Print($"  Stack: {ex.StackTrace}");
                }
                
                LogDebug("Configure complete");
            }
            else if (State == State.DataLoaded)
            {
                LogDebug("Datos cargados");
            }
            else if (State == State.Historical)
            {
                LogDebug("Iniciando procesamiento histórico");
            }
            else if (State == State.Transition)
            {
                LogDebug("En transición a tiempo real");
            }
            else if (State == State.Realtime)
            {
                LogDebug("Iniciando procesamiento en tiempo real");
                
                // Mostrar instrucciones de exportación si está habilitado
                if (ButtonExport)
                {
                    try
                    {
                        this.Dispatcher.InvokeAsync(() =>
                        {
                            try 
                            {
                                // En lugar de un botón físico, mostrar instrucciones en el gráfico
                                Draw.TextFixed(this, "ExportInstructions", 
                                              "Para exportar datos, actualice el indicador o espere a que se complete automáticamente", 
                                              TextPosition.TopRight);
                                LogDebug("Instrucciones de exportación añadidas al gráfico");
                            }
                            catch (Exception uiEx)
                            {
                                Print($"Error al agregar instrucciones de exportación: {uiEx.Message}");
                            }
                        });
                    }
                    catch (Exception ex)
                    {
                        Print($"Error al mostrar instrucciones de exportación: {ex.Message}");
                    }
                }
                
                // Si está habilitado, exportar inmediatamente al llegar a tiempo real
                if (ForceExportOnRealtime && ExportToCSV && !hasExecutedExport)
                {
                    Print("Exportación forzada iniciada al llegar a tiempo real");
                    ExportDataToCSV();
                    hasExecutedExport = true;
                }
            }
            else if (State == State.Terminated)
            {
                LogDebug("Indicador terminado");
                
                // Limpiar instrucciones de exportación
                if (ButtonExport)
                {
                    try
                    {
                        this.Dispatcher.InvokeAsync(() =>
                        {
                            try
                            {
                                // Limpiar cualquier texto fijo que hayamos agregado
                                Draw.TextFixed(this, "ExportInstructions", string.Empty, TextPosition.TopRight);
                                LogDebug("Instrucciones de exportación removidas del gráfico");
                            }
                            catch (Exception ex)
                            {
                                Print($"Error al limpiar instrucciones: {ex.Message}");
                            }
                        });
                    }
                    catch (Exception) { }
                }
            }
        }

        protected override void OnBarUpdate()
        {
            try
            {
                // Solo muestra mensajes cada 100 barras para no sobrecargar el log
                if (DebugMode && CurrentBar % 100 == 0)
                {
                    LogDebug($"OnBarUpdate - Barra #{CurrentBar}, BarsInProgress={BarsInProgress}, IsFirstTickOfBar={IsFirstTickOfBar}");
                }
                
                // Actualizar componentes ADX manualmente
                if (BarsInProgress == 0 && CurrentBar >= 1)
                {
                    try
                    {
                        // Computo manual de la línea +DI y -DI usando Formula DI
                        double trueRange = Math.Max(High[0] - Low[0], Math.Max(Math.Abs(High[0] - Close[1]), Math.Abs(Low[0] - Close[1])));
                        
                        if (trueRange > 0)
                        {
                            double upMove = High[0] - High[1];
                            double downMove = Low[1] - Low[0];
                            
                            double plusDM = upMove > downMove && upMove > 0 ? upMove : 0;
                            double minusDM = downMove > upMove && downMove > 0 ? downMove : 0;
                            
                            // Almacenar los valores en nuestras propias series
                            adxPlus[0] = (plusDM / trueRange) * 100;
                            adxMinus[0] = (minusDM / trueRange) * 100;
                        }
                        else
                        {
                            // Evitar división por cero
                            adxPlus[0] = adxPlus[1];  // Mantener valor anterior
                            adxMinus[0] = adxMinus[1];  // Mantener valor anterior
                        }
                    }
                    catch (Exception ex)
                    {
                        LogDebug($"Error calculando ADX components: {ex.Message}");
                    }
                }
                
                // Marcar cuando terminamos de procesar los datos históricos
                if (State == State.Historical && BarsInProgress == 0 && CurrentBar > 0 && !historyProcessed)
                {
                    historyProcessed = true;
                    LogDebug("Procesamiento histórico completado");
                }
                
                // Verificar condiciones para exportar
                if (ExportToCSV && !hasExecutedExport && 
                    (
                        IsLastBarOnChart() || 
                        (CurrentBar >= ExportDelayBars && ExportDelayBars > 0) ||
                        (State == State.Realtime && !hasExecutedExport && !ForceExportOnRealtime)
                    ))
                {
                    LogDebug($"Preparando para exportar datos: IsLastBarOnChart={IsLastBarOnChart()}, CurrentBar={CurrentBar}");
                    Print($"EXPORTACIÓN INICIADA: Condiciones - CurrentBar={CurrentBar}, ExportDelayBars={ExportDelayBars}, State={State}");
                    ExportDataToCSV();
                    hasExecutedExport = true;
                    LogDebug("Exportación de datos completada");
                }
            }
            catch (Exception ex)
            {
                Print($"ERROR en OnBarUpdate: {ex.Message}");
            }
        }

        private void ExportDataToCSV()
        {
            Print("==== INICIANDO EXPORTACIÓN DE DATOS A CSV ====");
            LogDebug("Iniciando ExportDataToCSV...");
            
            try
            {
                if (ChartControl == null)
                {
                    Print("ERROR: ChartControl no está disponible, no se puede exportar");
                    return;
                }

                // Obtener y sanitizar el nombre del instrumento
                string instrumentName = Instrument.FullName;
                string sanitizedInstrument = SanitizeFileName(instrumentName);
                LogDebug($"Instrumento: {instrumentName}, sanitizado: {sanitizedInstrument}");

                // Obtener fecha y hora actuales
                DateTime now = DateTime.Now;
                string datePart = now.ToString("yyyyMMdd"); 
                string timePart = now.ToString("HHmmss"); 

                // Construir el nombre del archivo con formato compatible con Windows
                string fileName = $"{sanitizedInstrument}_Data_{datePart}_{timePart}.csv";
                LogDebug($"Nombre de archivo generado: {fileName}");

                // Usar la ruta especificada por el usuario
                string exportFolder = ExportPath;
                
                // Verificar si el directorio existe, si no, crearlo
                if (!Directory.Exists(exportFolder))
                {
                    try
                    {
                        LogDebug($"Creando directorio: {exportFolder}");
                        Directory.CreateDirectory(exportFolder);
                        LogDebug($"Directorio creado: {exportFolder}");
                    }
                    catch (Exception ex)
                    {
                        Print($"ERROR: No se pudo crear la carpeta {exportFolder}: {ex.Message}");
                        
                        // Intentar usar directorio temporal como alternativa
                        string tempPath = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
                        Print($"Intentando usar escritorio como alternativa: {tempPath}");
                        
                        if (Directory.Exists(tempPath))
                        {
                            exportFolder = tempPath;
                            Print($"Usando escritorio como directorio alternativo: {exportFolder}");
                        }
                        else
                        {
                            // Última opción: directorio temporal del sistema
                            exportFolder = Path.GetTempPath();
                            Print($"Usando directorio temporal del sistema: {exportFolder}");
                        }
                    }
                }
                else
                {
                    LogDebug($"Directorio existe: {exportFolder}");
                    // Verificar permisos con una prueba rápida
                    try
                    {
                        string testFile = Path.Combine(exportFolder, "test_permissions.tmp");
                        File.WriteAllText(testFile, "test");
                        File.Delete(testFile);
                        LogDebug("Prueba de permisos de escritura exitosa");
                    }
                    catch (Exception ex)
                    {
                        Print($"ERROR: No hay permisos de escritura en {exportFolder}: {ex.Message}");
                        exportFolder = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
                        Print($"Cambiando al escritorio: {exportFolder}");
                    }
                }

                // Ruta completa del archivo
                string fullPath = Path.Combine(exportFolder, fileName);
                Print($"Ruta completa del archivo de exportación: {fullPath}");
                
                // Verificar que la ruta sea válida
                try
                {
                    FileInfo fileInfo = new FileInfo(fullPath);
                    LogDebug($"Ruta del archivo validada: {fullPath}");
                }
                catch (Exception ex)
                {
                    Print($"ERROR en la ruta del archivo: {ex.Message}");
                    fullPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "export_data.csv");
                    Print($"Usando ruta alternativa: {fullPath}");
                }

                // Obtener la lista de otros indicadores en el gráfico
                LogDebug("Recuperando indicadores del gráfico...");
                var otherIndicators = new List<Indicator>();
                
                try
                {
                    if (ChartControl != null && ChartControl.Indicators != null)
                    {
                        // Corregido: usar Cast<Indicator>() para convertir el tipo apropiadamente
                        var chartIndicators = ChartControl.Indicators.Where(ind => ind != this);
                        foreach (var indicator in chartIndicators)
                        {
                            // Intentar convertir cada indicador individualmente
                            if (indicator is Indicator)
                                otherIndicators.Add(indicator as Indicator);
                        }
                        LogDebug($"Encontrados {otherIndicators.Count} indicadores adicionales en el gráfico");
                    }
                    else
                    {
                        LogDebug("No se pudieron recuperar indicadores del gráfico");
                    }
                }
                catch (Exception ex)
                {
                    Print($"ERROR recuperando indicadores: {ex.Message}");
                }

                // Construir el encabezado del CSV con indicadores específicos para detección de patrones
                List<string> header = new List<string> 
                { 
                    "Timestamp", "Open", "High", "Low", "Close", "Volume", 
                    "FastEMA", "SlowEMA", "RSI", "Stoch_K", "Stoch_D", 
                    "BB_Upper", "BB_Middle", "BB_Lower", "MACD", "MACD_Signal", "MACD_Hist",
                    "ATR", "ADX", "DI_Plus", "DI_Minus", "EMACrossover", "PriceToSlowEMA",
                    "BarType", "SwingHigh", "SwingLow", "RangePercent"
                };
                
                // Procesar indicadores adicionales para el encabezado
                LogDebug("Procesando indicadores adicionales para encabezado...");
                var extraIndicatorPlots = new List<Tuple<string, Indicator, int>>();
                
                foreach (var ind in otherIndicators)
                {
                    try
                    {
                        var plots = ind.Plots.ToList();
                        LogDebug($"Indicador: {ind.Name} tiene {plots.Count} plots");
                        
                        for (int plotIndex = 0; plotIndex < plots.Count; plotIndex++)
                        {
                            var plot = plots[plotIndex];
                            string plotName = plot.Name;
                            string headerName = $"{ind.Name}_{plotName}";
                            header.Add(headerName);
                            
                            // Guardar referencia al indicador y su índice de plot
                            extraIndicatorPlots.Add(Tuple.Create(headerName, ind, plotIndex));
                        }
                    }
                    catch (Exception ex)
                    {
                        Print($"ERROR al procesar indicador {ind.Name}: {ex.Message}");
                        string headerName = $"{ind.Name}_Value";
                        header.Add(headerName);
                    }
                }

                // Obtener el número total de barras y limitar según la configuración
                int totalBarsInChart = Bars.Count;
                Print($"Total de barras disponibles: {totalBarsInChart}");
                
                if (totalBarsInChart <= 0)
                {
                    totalBarsInChart = CurrentBar + 1;
                    Print($"Usando CurrentBar+1 como total: {totalBarsInChart}");
                }
                
                // Limitar al máximo especificado por el usuario
                int barsToExport = Math.Min(totalBarsInChart, MaxBarsToExport);
                Print($"Barras a exportar: {barsToExport}");

                // Calcular el índice inicial y final para la exportación (desde el más antiguo al más reciente)
                // Este enfoque soluciona el problema de la limitación de barsAgo
                int startIndex = Math.Max(0, totalBarsInChart - barsToExport);
                int endIndex = totalBarsInChart - 1;
                
                Print($"Exportando desde índice {startIndex} hasta {endIndex} (total: {endIndex - startIndex + 1} barras)");
                
                // Lista para almacenar todas las filas en orden cronológico
                List<string> csvRows = new List<string>();
                
                // Añadir encabezado como primera fila
                csvRows.Add(string.Join(",", header));
                int successfulRows = 0;

                try
                {
                    // Procesar barras desde la más antigua a la más reciente (orden cronológico)
                    for (int barIndex = startIndex; barIndex <= endIndex; barIndex++)
                    {
                        List<string> row = new List<string>();
                        
                        try
                        {
                            // Calcular indiceBarsArray como índice relativo a la serie de datos
                            int indiceBarsArray = barIndex;
                            
                            // Datos OHLCV básicos usando GetValueAt que es más robusto que el operador []
                            row.Add(Time.GetValueAt(indiceBarsArray).ToString("yyyy-MM-dd HH:mm:ss"));
                            row.Add(Open.GetValueAt(indiceBarsArray).ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(High.GetValueAt(indiceBarsArray).ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(Low.GetValueAt(indiceBarsArray).ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(Close.GetValueAt(indiceBarsArray).ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(volume.GetValueAt(indiceBarsArray).ToString(System.Globalization.CultureInfo.InvariantCulture));
                            
                            // Indicadores técnicos usando GetValueAt
                            row.Add(fastEMA.GetValueAt(indiceBarsArray).ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(slowEMA.GetValueAt(indiceBarsArray).ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(rsi.GetValueAt(indiceBarsArray).ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(stoch.K.GetValueAt(indiceBarsArray).ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(stoch.D.GetValueAt(indiceBarsArray).ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(bbands.Upper.GetValueAt(indiceBarsArray).ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(bbands.Middle.GetValueAt(indiceBarsArray).ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(bbands.Lower.GetValueAt(indiceBarsArray).ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(macd.Default.GetValueAt(indiceBarsArray).ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(macd.Avg.GetValueAt(indiceBarsArray).ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(macd.Diff.GetValueAt(indiceBarsArray).ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(atr.GetValueAt(indiceBarsArray).ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(adx.GetValueAt(indiceBarsArray).ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(adxPlus.GetValueAt(indiceBarsArray).ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(adxMinus.GetValueAt(indiceBarsArray).ToString(System.Globalization.CultureInfo.InvariantCulture));
                            
                            // Características adicionales para detección de patrones
                            // EMACrossover: 1 cuando FastEMA cruza por encima de SlowEMA, -1 cuando cruza por debajo, 0 en otro caso
                            double emaCrossover = 0;
                            if (indiceBarsArray > 0 && indiceBarsArray < totalBarsInChart - 1)
                            {
                                double fastEMAcurrent = fastEMA.GetValueAt(indiceBarsArray);
                                double slowEMAcurrent = slowEMA.GetValueAt(indiceBarsArray);
                                double fastEMAprevious = fastEMA.GetValueAt(indiceBarsArray - 1);
                                double slowEMAprevious = slowEMA.GetValueAt(indiceBarsArray - 1);
                                
                                if (fastEMAcurrent > slowEMAcurrent && fastEMAprevious <= slowEMAprevious)
                                    emaCrossover = 1;
                                else if (fastEMAcurrent < slowEMAcurrent && fastEMAprevious >= slowEMAprevious)
                                    emaCrossover = -1;
                            }
                            row.Add(emaCrossover.ToString(System.Globalization.CultureInfo.InvariantCulture));
                            
                            // PriceToSlowEMA: Diferencia porcentual entre precio de cierre y EMA lenta
                            double closePrice = Close.GetValueAt(indiceBarsArray);
                            double slowEMAvalue = slowEMA.GetValueAt(indiceBarsArray);
                            double priceToEma = (closePrice - slowEMAvalue) / slowEMAvalue;
                            row.Add(priceToEma.ToString(System.Globalization.CultureInfo.InvariantCulture));
                            
                            // BarType: 1 para alcista, -1 para bajista, 0 para doji
                            double barType = 0;
                            double openPrice = Open.GetValueAt(indiceBarsArray);
                            if (closePrice > openPrice)
                                barType = 1;
                            else if (closePrice < openPrice)
                                barType = -1;
                            row.Add(barType.ToString(System.Globalization.CultureInfo.InvariantCulture));
                            
                            // Detectar Swing High - Máximo local
                            bool isSwingHigh = false;
                            if (indiceBarsArray > 2 && indiceBarsArray < totalBarsInChart - 2)
                            {
                                double highValue = High.GetValueAt(indiceBarsArray);
                                double highPrev1 = High.GetValueAt(indiceBarsArray - 1);
                                double highNext1 = High.GetValueAt(indiceBarsArray + 1);
                                double highNext2 = High.GetValueAt(indiceBarsArray + 2);
                                
                                isSwingHigh = highValue > highPrev1 && highValue > highNext1 && highValue > highNext2;
                            }
                            row.Add(isSwingHigh ? "1" : "0");
                            
                            // Detectar Swing Low - Mínimo local
                            bool isSwingLow = false;
                            if (indiceBarsArray > 2 && indiceBarsArray < totalBarsInChart - 2)
                            {
                                double lowValue = Low.GetValueAt(indiceBarsArray);
                                double lowPrev1 = Low.GetValueAt(indiceBarsArray - 1);
                                double lowNext1 = Low.GetValueAt(indiceBarsArray + 1);
                                double lowNext2 = Low.GetValueAt(indiceBarsArray + 2);
                                
                                isSwingLow = lowValue < lowPrev1 && lowValue < lowNext1 && lowValue < lowNext2;
                            }
                            row.Add(isSwingLow ? "1" : "0");
                            
                            // RangePercent - Tamaño del rango de la vela como porcentaje
                            double highPrice = High.GetValueAt(indiceBarsArray);
                            double lowPrice = Low.GetValueAt(indiceBarsArray);
                            double rangePercent = (highPrice - lowPrice) / lowPrice;
                            row.Add(rangePercent.ToString(System.Globalization.CultureInfo.InvariantCulture));
                            
                            // Añadir valores de indicadores adicionales
                            foreach (var indicatorInfo in extraIndicatorPlots)
                            {
                                try 
                                {
                                    string headerName = indicatorInfo.Item1;
                                    Indicator indicator = indicatorInfo.Item2;
                                    int plotIndex = indicatorInfo.Item3;
                                    
                                    // Obtener la serie de valores del indicador
                                    ISeries<double> series = indicator.Values[plotIndex] as ISeries<double>;
                                    
                                    if (series != null && indiceBarsArray < series.Count)
                                    {
                                        double value = series.GetValueAt(indiceBarsArray);
                                        row.Add(value.ToString(System.Globalization.CultureInfo.InvariantCulture));
                                    }
                                    else
                                    {
                                        row.Add(string.Empty);
                                    }
                                }
                                catch (Exception ex)
                                {
                                    Print($"Error al obtener valor de indicador adicional: {ex.Message}");
                                    row.Add(string.Empty);
                                }
                            }
                            
                            csvRows.Add(string.Join(",", row));
                            successfulRows++;
                            
                            // Mostrar progreso cada 1000 barras
                            if (successfulRows % 1000 == 0)
                            {
                                Print($"Procesadas {successfulRows} barras de {barsToExport}...");
                            }
                        }
                        catch (Exception ex)
                        {
                            Print($"ERROR al procesar barra con índice={barIndex}: {ex.Message}");
                            continue; // Continuar con la siguiente barra
                        }
                    }

                    LogDebug($"Escribiendo {csvRows.Count} filas al archivo CSV...");
                    
                    // Escribir todas las filas al archivo
                    File.WriteAllLines(fullPath, csvRows);
                    
                    Print($"¡EXPORTACIÓN COMPLETADA! Datos exportados: {successfulRows} barras a {fullPath}");
                    
                    try {
                        Draw.TextFixed(this, "ExportStatus", $"Datos exportados: {successfulRows} barras\nRuta: {fullPath}", TextPosition.BottomRight);
                    } catch (Exception ex) {
                        Print($"ERROR al mostrar mensaje en gráfico: {ex.Message}");
                    }
                }
                catch (Exception ex)
                {
                    Print($"ERROR al escribir en el archivo CSV: {ex.Message}");
                    if (ex.InnerException != null)
                    {
                        Print($"Inner Exception: {ex.InnerException.Message}");
                    }
                }
            }
            catch (Exception ex)
            {
                Print($"ERROR CRÍTICO al exportar datos: {ex.Message}");
                if (ex.InnerException != null)
                {
                    Print($"Inner Exception: {ex.InnerException.Message}");
                }
                Print($"Stack Trace: {ex.StackTrace}");
                
                try {
                    Draw.TextFixed(this, "ExportError", $"ERROR: {ex.Message}", TextPosition.BottomRight);
                } catch (Exception drawEx) {
                    Print($"ERROR al mostrar mensaje de error en gráfico: {drawEx.Message}");
                }
            }
        }

        private string SanitizeFileName(string name)
        {
            foreach (char c in Path.GetInvalidFileNameChars())
            {
                name = name.Replace(c, '_');
            }
            
            name = name.Replace(' ', '_')
                       .Replace(':', '_').Replace('/', '_').Replace('\\', '_')
                       .Replace('[', '_').Replace(']', '_').Replace('{', '_').Replace('}', '_')
                       .Replace('(', '_').Replace(')', '_').Replace('*', '_').Replace('?', '_');
            
            return name;
        }
        
        private bool IsLastBarOnChart()
        {
            // Mejora de la detección de última barra
            bool isLast = BarsInProgress == 0 && 
                          CurrentBars[0] >= 0 && 
                          (CurrentBars[0] >= BarsArray[0].Count - 5) && // Más permisivo, considera últimas 5 barras
                          IsFirstTickOfBar;
            
            if (isLast)
            {
                Print($"Detectada última barra: CurrentBars[0]={CurrentBars[0]}, BarsArray[0].Count={BarsArray[0].Count}");
            }
            
            return isLast;
        }
        
        private void LogDebug(string message)
        {
            if (DebugMode)
            {
                Print($"[DEBUG] AITraderDataExtractor: {message}");
                
                // También mostrar en el gráfico para los mensajes importantes
                if (message.Contains("ERROR") || message.Contains("export") || 
                    message.StartsWith("Iniciando") || message.Contains("completada"))
                {
                    debugCounter++;
                    if (debugCounter <= 10) // Limitar número de mensajes en el gráfico
                    {
                        try {
                            Draw.TextFixed(this, $"Debug_{debugCounter}", message, TextPosition.TopRight);
                        } catch (Exception ex) {
                            Print($"ERROR al mostrar mensaje de debug en gráfico: {ex.Message}");
                        }
                    }
                }
            }
        }
    }
}