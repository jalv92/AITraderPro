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
        [Display(Name = "Max Bars to Export", Description = "Maximum number of bars to export (from newest to oldest)", Order = 3, GroupName = "Parameters")]
        public int MaxBarsToExport { get; set; }
        
        [NinjaScriptProperty]
        [Range(10, 200)]
        [Display(Name = "Fast EMA Period", Description = "Period for Fast EMA", Order = 4, GroupName = "Indicators")]
        public int FastEMAPeriod { get; set; }
        
        [NinjaScriptProperty]
        [Range(20, 300)]
        [Display(Name = "Slow EMA Period", Description = "Period for Slow EMA", Order = 5, GroupName = "Indicators")]
        public int SlowEMAPeriod { get; set; }
        
        [NinjaScriptProperty]
        [Range(5, 50)]
        [Display(Name = "RSI Period", Description = "Period for RSI indicator", Order = 6, GroupName = "Indicators")]
        public int RSIPeriod { get; set; }
        
        [NinjaScriptProperty]
        [Range(5, 50)]
        [Display(Name = "Stochastic Period", Description = "Period for Stochastic indicator", Order = 7, GroupName = "Indicators")]
        public int StochasticPeriod { get; set; }
        
        [NinjaScriptProperty]
        [Range(10, 50)]
        [Display(Name = "Bollinger Bands Period", Description = "Period for Bollinger Bands", Order = 8, GroupName = "Indicators")]
        public int BollingerPeriod { get; set; }
        
        [NinjaScriptProperty]
        [Range(1.0, 3.0)]
        [Display(Name = "Bollinger Bands StdDev", Description = "Standard deviation for Bollinger Bands", Order = 9, GroupName = "Indicators")]
        public double BollingerStdDev { get; set; }
        
        [NinjaScriptProperty]
        [Range(5, 50)]
        [Display(Name = "MACD Fast", Description = "Fast period for MACD", Order = 10, GroupName = "Indicators")]
        public int MACDFast { get; set; }
        
        [NinjaScriptProperty]
        [Range(10, 100)]
        [Display(Name = "MACD Slow", Description = "Slow period for MACD", Order = 11, GroupName = "Indicators")]
        public int MACDSlow { get; set; }
        
        [NinjaScriptProperty]
        [Range(3, 20)]
        [Display(Name = "MACD Signal", Description = "Signal period for MACD", Order = 12, GroupName = "Indicators")]
        public int MACDSignal { get; set; }
        
        [NinjaScriptProperty]
        [Range(7, 100)]
        [Display(Name = "ATR Period", Description = "Period for Average True Range", Order = 13, GroupName = "Indicators")]
        public int ATRPeriod { get; set; }
        
        [NinjaScriptProperty]
        [Range(3, 20)]
        [Display(Name = "ADX Period", Description = "Period for ADX", Order = 14, GroupName = "Indicators")]
        public int ADXPeriod { get; set; }
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
                MaxBarsToExport = 1000;
                
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
            }
            else if (State == State.Configure)
            {
                // Añadir indicadores
                fastEMA = EMA(FastEMAPeriod);
                slowEMA = EMA(SlowEMAPeriod);
                rsi = RSI(RSIPeriod, 1);
                stoch = Stochastics(StochasticPeriod, 3, 3);
                bbands = Bollinger(Convert.ToInt32(BollingerPeriod), Convert.ToInt32(BollingerStdDev));
                macd = MACD(MACDFast, MACDSlow, MACDSignal);
                atr = ATR(ATRPeriod);
                adx = ADX(ADXPeriod);
                volume = VOL();
                
                // Crear series personalizadas para los componentes de ADX
                adxPlus = new Series<double>(this);
                adxMinus = new Series<double>(this);
            }
        }

        protected override void OnBarUpdate()
        {
            // Actualizar componentes ADX manualmente
            if (BarsInProgress == 0)
            {
                // Computo manual de la línea +DI y -DI usando Formula DI
                if (CurrentBar >= 1)
                {
                    // Estas son aproximaciones simplificadas. Para mayor precisión, consulta la documentación de NinjaTrader
                    double trueRange = Math.Max(High[0] - Low[0], Math.Max(Math.Abs(High[0] - Close[1]), Math.Abs(Low[0] - Close[1])));
                    double upMove = High[0] - High[1];
                    double downMove = Low[1] - Low[0];
                    
                    double plusDM = upMove > downMove && upMove > 0 ? upMove : 0;
                    double minusDM = downMove > upMove && downMove > 0 ? downMove : 0;
                    
                    // Almacenar los valores en nuestras propias series
                    adxPlus[0] = (plusDM / trueRange) * 100;
                    adxMinus[0] = (minusDM / trueRange) * 100;
                }
            }
            
            if (ExportToCSV && CurrentBar > 0 && IsLastBarOnChart())
            {
                ExportDataToCSV();
                ExportToCSV = false;
            }
        }

        private void ExportDataToCSV()
        {
            try
            {
                if (ChartControl == null)
                {
                    Print("ChartControl no está disponible, no se puede exportar");
                    return;
                }

                // Obtener y sanitizar el nombre del instrumento
                string instrumentName = Instrument.FullName;
                string sanitizedInstrument = SanitizeFileName(instrumentName);

                // Obtener fecha y hora actuales
                DateTime now = DateTime.Now;
                string datePart = now.ToString("MM-yyyy"); 
                string timePart = now.ToString("HHmm"); 

                // Construir el nombre del archivo con formato compatible con Windows
                string fileName = $"{sanitizedInstrument}_Data_{datePart}_{timePart}.csv";

                // Usar la ruta especificada por el usuario
                string exportFolder = ExportPath;
                if (!Directory.Exists(exportFolder))
                {
                    try
                    {
                        Directory.CreateDirectory(exportFolder);
                    }
                    catch (Exception ex)
                    {
                        Print($"No se pudo crear la carpeta: {ex.Message}");
                        exportFolder = Path.GetTempPath();
                        Print($"Usando directorio temporal: {exportFolder}");
                    }
                }

                // Ruta completa del archivo
                string fullPath = Path.Combine(exportFolder, fileName);
                
                // Verificar que la ruta sea válida
                try
                {
                    FileInfo fileInfo = new FileInfo(fullPath);
                    Print($"Ruta del archivo validada: {fullPath}");
                }
                catch (Exception ex)
                {
                    Print($"Error en la ruta del archivo: {ex.Message}");
                    fullPath = Path.Combine(Path.GetTempPath(), "export_data.csv");
                    Print($"Usando ruta alternativa: {fullPath}");
                }

                // Obtener la lista de otros indicadores en el gráfico
                var otherIndicators = ChartControl.Indicators.Where(ind => ind != this).ToList();

                // Crear mapeo de indicadores a sus valores
                var indicatorValues = new List<Tuple<string, Func<int, string>>>();

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
                foreach (var ind in otherIndicators)
                {
                    try
                    {
                        var plots = ind.Plots.ToList();
                        for (int plotIndex = 0; plotIndex < plots.Count; plotIndex++)
                        {
                            var plot = plots[plotIndex];
                            string plotName = plot.Name;
                            string headerName = $"{ind.Name}_{plotName}";
                            header.Add(headerName);
                            
                            int finalPlotIndex = plotIndex;
                            indicatorValues.Add(Tuple.Create(headerName, (Func<int, string>)(barsAgo => {
                                try
                                {
                                    if (barsAgo < 0 || barsAgo >= BarsArray[0].Count)
                                        return string.Empty;
                                        
                                    double value = ((ISeries<double>)ind.Values[finalPlotIndex])[barsAgo];
                                    return value.ToString(System.Globalization.CultureInfo.InvariantCulture);
                                }
                                catch
                                {
                                    return string.Empty;
                                }
                            })));
                        }
                    }
                    catch (Exception ex)
                    {
                        Print($"Error al procesar indicador {ind.Name}: {ex.Message}");
                        string headerName = $"{ind.Name}_Value";
                        header.Add(headerName);
                        indicatorValues.Add(Tuple.Create(headerName, (Func<int, string>)(_ => string.Empty)));
                    }
                }

                // Obtener el número total de barras
                int totalBarsInChart = 0;
                
                try
                {
                    // Intentar obtener el conteo de barras desde ChartBars
                    if (ChartControl != null && ChartControl.ChartPanels[0].ChartObjects != null)
                    {
                        foreach (var obj in ChartControl.ChartPanels[0].ChartObjects)
                        {
                            if (obj is ChartBars chartBars)
                            {
                                totalBarsInChart = chartBars.Count;
                                break;
                            }
                        }
                    }
                    
                    // Si no se obtiene de ChartBars, usar Bars.Count
                    if (totalBarsInChart <= 0)
                    {
                        totalBarsInChart = Bars.Count;
                    }
                    
                    // Si aún no hay valor válido, usar BarsArray
                    if (totalBarsInChart <= 0)
                    {
                        totalBarsInChart = BarsArray[0].Count;
                    }
                    
                    // Última opción: usar un valor por defecto
                    if (totalBarsInChart <= 0)
                    {
                        totalBarsInChart = 1000;
                        Print("No se pudo determinar el número exacto de barras, usando valor por defecto: 1000");
                    }
                }
                catch (Exception ex)
                {
                    Print($"Error al obtener el total de barras: {ex.Message}");
                    totalBarsInChart = 1000; // Valor por defecto si ocurre un error
                }
                
                // Limitar al máximo especificado por el usuario
                int barsToExport = Math.Min(totalBarsInChart, MaxBarsToExport);
                
                Print($"Barras totales en el gráfico: {totalBarsInChart}, Barras a exportar: {barsToExport}");

                // Lista para almacenar las filas
                List<string> allRows = new List<string>();

                // Escribir en el archivo CSV
                using (StreamWriter writer = new StreamWriter(fullPath))
                {
                    writer.WriteLine(string.Join(",", header));

                    // Procesar barras desde la más reciente hacia atrás
                    for (int i = 0; i < barsToExport; i++)
                    {
                        int barsAgo = Math.Min(i, CurrentBar);
                        
                        if (barsAgo >= BarsArray[0].Count)
                            break;

                        List<string> row = new List<string>();
                        
                        try
                        {
                            // Datos OHLCV básicos
                            row.Add(Time[barsAgo].ToString("yyyy-MM-dd HH:mm:ss"));
                            row.Add(Open[barsAgo].ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(High[barsAgo].ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(Low[barsAgo].ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(Close[barsAgo].ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(volume[barsAgo].ToString(System.Globalization.CultureInfo.InvariantCulture));
                            
                            // Indicadores técnicos
                            row.Add(fastEMA[barsAgo].ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(slowEMA[barsAgo].ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(rsi[barsAgo].ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(stoch.K[barsAgo].ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(stoch.D[barsAgo].ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(bbands.Upper[barsAgo].ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(bbands.Middle[barsAgo].ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(bbands.Lower[barsAgo].ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(macd.Default[barsAgo].ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(macd.Avg[barsAgo].ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(macd.Diff[barsAgo].ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(atr[barsAgo].ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(adx[barsAgo].ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(adxPlus[barsAgo].ToString(System.Globalization.CultureInfo.InvariantCulture));
                            row.Add(adxMinus[barsAgo].ToString(System.Globalization.CultureInfo.InvariantCulture));
                            
                            // Características adicionales para detección de patrones
                            // EMACrossover: 1 cuando FastEMA cruza por encima de SlowEMA, -1 cuando cruza por debajo, 0 en otro caso
                            double emaCrossover = 0;
                            if (barsAgo < BarsArray[0].Count - 1)
                            {
                                if (fastEMA[barsAgo] > slowEMA[barsAgo] && fastEMA[barsAgo + 1] <= slowEMA[barsAgo + 1])
                                    emaCrossover = 1;
                                else if (fastEMA[barsAgo] < slowEMA[barsAgo] && fastEMA[barsAgo + 1] >= slowEMA[barsAgo + 1])
                                    emaCrossover = -1;
                            }
                            row.Add(emaCrossover.ToString(System.Globalization.CultureInfo.InvariantCulture));
                            
                            // PriceToSlowEMA: Diferencia porcentual entre precio de cierre y EMA lenta
                            double priceToEma = (Close[barsAgo] - slowEMA[barsAgo]) / slowEMA[barsAgo] * 100;
                            row.Add(priceToEma.ToString(System.Globalization.CultureInfo.InvariantCulture));
                            
                            // BarType: 1 para alcista, -1 para bajista, 0 para doji
                            double barType = 0;
                            if (Close[barsAgo] > Open[barsAgo])
                                barType = 1;
                            else if (Close[barsAgo] < Open[barsAgo])
                                barType = -1;
                            row.Add(barType.ToString(System.Globalization.CultureInfo.InvariantCulture));
                            
                            // Detectar Swing High - Máximo local
                            bool isSwingHigh = false;
                            if (barsAgo < BarsArray[0].Count - 2 && barsAgo > 0)
                            {
                                isSwingHigh = High[barsAgo] > High[barsAgo - 1] && High[barsAgo] > High[barsAgo + 1] && 
                                              High[barsAgo] > High[barsAgo + 2];
                            }
                            row.Add(isSwingHigh ? "1" : "0");
                            
                            // Detectar Swing Low - Mínimo local
                            bool isSwingLow = false;
                            if (barsAgo < BarsArray[0].Count - 2 && barsAgo > 0)
                            {
                                isSwingLow = Low[barsAgo] < Low[barsAgo - 1] && Low[barsAgo] < Low[barsAgo + 1] && 
                                             Low[barsAgo] < Low[barsAgo + 2];
                            }
                            row.Add(isSwingLow ? "1" : "0");
                            
                            // RangePercent - Tamaño del rango de la vela como porcentaje
                            double rangePercent = (High[barsAgo] - Low[barsAgo]) / Low[barsAgo] * 100;
                            row.Add(rangePercent.ToString(System.Globalization.CultureInfo.InvariantCulture));
                            
                            // Añadir valores de indicadores adicionales
                            foreach (var valueFunc in indicatorValues)
                            {
                                string value = valueFunc.Item2(barsAgo);
                                row.Add(value);
                            }
                            
                            allRows.Add(string.Join(",", row));
                        }
                        catch (Exception ex)
                        {
                            Print($"Error al procesar barra con barsAgo={barsAgo}: {ex.Message}");
                            break;
                        }
                    }

                    // Escribir filas en orden cronológico (invertido)
                    for (int i = allRows.Count - 1; i >= 0; i--)
                    {
                        writer.WriteLine(allRows[i]);
                    }
                }

                Print($"Datos exportados: {allRows.Count} barras a {fullPath}");
            }
            catch (Exception ex)
            {
                Print($"Error al exportar datos: {ex.Message}");
                if (ex.InnerException != null)
                {
                    Print($"Inner Exception: {ex.InnerException.Message}");
                }
                Print($"Stack Trace: {ex.StackTrace}");
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
            return BarsInProgress == 0 && CurrentBars[0] >= 0 && CurrentBars[0] == BarsArray[0].Count - 1 && IsFirstTickOfBar;
        }
    }
}